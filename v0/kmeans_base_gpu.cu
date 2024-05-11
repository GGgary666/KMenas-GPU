#include <cmath>
#include <iostream>
#include "error.cuh"
#include "kmeans_base_gpu.cuh"

KmeansGPU::KmeansGPU(int numClusters, int numFeatures, float *clusters, int nsamples) : Kmeans(numClusters, numFeatures, clusters, nsamples) {}

KmeansGPU::KmeansGPU(int numClusters, int numFeatures, float *clusters, int nsamples,
                     int maxIters, float epsilon) : Kmeans(numClusters, numFeatures, clusters, nsamples,
                                                           maxIters, epsilon) {}



template <typename T>
__global__ void init(T *x, const T value, const int N)
{
    const int n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n < N)
        x[n] = value;
}

__global__ void calDistKernel(
    const float *d_data,
    const float *d_clusters, // [numClusters, numFeatures]
    float *d_distance,       // [nsamples, numClusters]
    const int numClusters,
    const int clusterNo,
    const int nsamples,
    const int numFeatures)
{

    int n = threadIdx.x + numFeatures * blockIdx.x;
    int m = threadIdx.x + numFeatures * clusterNo;
    extern __shared__ float s_c[];
    s_c[threadIdx.x] = 0.0;
    if (n < numFeatures * nsamples && threadIdx.x < numFeatures)
    {
        s_c[threadIdx.x] = powf(d_data[n] - d_clusters[m], 2);
    }
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (threadIdx.x < offset)
            s_c[threadIdx.x] += s_c[threadIdx.x + offset];
        __syncthreads();
    }
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
            s_c[threadIdx.x] += s_c[threadIdx.x + offset];
        __syncwarp();
    }
    if (threadIdx.x == 0)
        d_distance[blockIdx.x * numClusters + clusterNo] = sqrt(s_c[0]);
}

__global__ void reduceMin(
    float *d_distance,
    int *d_sampleClasses,
    int *d_clusterSize,
    int numClusters,
    int nsamples,
    float *d_minDist)
{
    int n = threadIdx.x + blockDim.x * blockIdx.x;
    if (n < nsamples)
    {
        float minDist = d_distance[n * numClusters + 0];
        int minIdx = 0;
        float tmp;
        for (int i = 1; i < numClusters; i++)
        {
            tmp = __ldg(&d_distance[n * numClusters + i]);
            if (tmp < minDist)
            {
                minDist = tmp;
                minIdx = i;
            }
        }
        d_sampleClasses[n] = minIdx;
        d_minDist[n] = minDist;
    }
}
__global__ void reduceSum(
    float *d_minDist,
    float *d_loss,
    int nsamples)
{
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    extern __shared__ float s_y[];
    float y = 0.0;
    const int stride = blockDim.x * gridDim.x;
    for (; n < nsamples; n += stride)
        y += d_minDist[n];
    s_y[threadIdx.x] = y;
    __syncthreads();
    for (int offset = blockDim.x >> 1; offset >= 32; offset >>= 1)
    {
        if (threadIdx.x < offset)
            s_y[threadIdx.x] += s_y[threadIdx.x + offset];
        __syncthreads();
    }
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
            s_y[threadIdx.x] += s_y[threadIdx.x + offset];
        __syncwarp();
    }
    if (threadIdx.x == 0)
        d_loss[blockIdx.x] = s_y[0];
}

__global__ void countCluster(int *d_sampleClasses, int *d_clusterSize, int nsamples)
{
    int n = threadIdx.x + blockDim.x * blockIdx.x;
    if (n < nsamples)
    {
        int clusterID = d_sampleClasses[n];
        atomicAdd(&(d_clusterSize[clusterID]), 1);
    }
}

__global__ void update(
    const float *d_data,
    float *d_clusters,
    int *d_sampleClasses,
    int *d_clusterSize,
    const int nsamples,
    const int numFeatures)
{

    int n = threadIdx.x + numFeatures * blockIdx.x;
    int clusterId = d_sampleClasses[blockIdx.x];
    int clustercnt = d_clusterSize[clusterId];
    if (threadIdx.x < numFeatures)
    {
        atomicAdd(&(d_clusters[clusterId * numFeatures + threadIdx.x]), d_data[n] / clustercnt);
    }
}

void updateClusterWithCuda(
    const float *d_data,
    float *d_clusters,
    int *d_sampleClasses,
    int *d_clusterSize,
    const int nsamples,
    const int numClusters,
    const int numFeatures)
{

    init<float><<<1, 1024>>>(d_clusters, 0.0, numClusters * numFeatures);
    int blockSize = 1024;
    int gridSize = (nsamples - 1) / blockSize + 1;
    countCluster<<<gridSize, blockSize>>>(d_sampleClasses, d_clusterSize, nsamples);
    update<<<nsamples, 128>>>(d_data, d_clusters, d_sampleClasses, d_clusterSize, nsamples, numFeatures);
}

void calDistWithCuda(
    const float *d_data,
    float *d_clusters,
    float *d_distance,
    int *d_sampleClasses,
    float *d_minDist,
    float *d_loss,
    int *d_clusterSize,
    const int numClusters,
    const int nsamples,
    const int numFeatures)
{

    init<int><<<1, 128>>>(d_clusterSize, 0, numClusters);
    int smem = sizeof(float) * 128;
    cudaStream_t streams[20];
    for (int i = 0; i < numClusters; i++)
    {
        CHECK(cudaStreamCreate(&(streams[i])));
    }
    for (int i = 0; i < numClusters; i++)
    {
        calDistKernel<<<nsamples, 128, smem, streams[i]>>>(d_data, d_clusters,
                                                           d_distance, numClusters, i, nsamples, numFeatures);
    }
    for (int i = 0; i < numClusters; ++i)
    {
        CHECK(cudaStreamDestroy(streams[i]));
    }
    int blockSize = 256;
    int gridSize = (nsamples - 1) / blockSize + 1;
    reduceMin<<<gridSize, blockSize, sizeof(int) * blockSize>>>(d_distance, d_sampleClasses,
                                                                d_clusterSize, numClusters, nsamples, d_minDist);
    reduceSum<<<256, 256, sizeof(float) * 256>>>(d_minDist, d_loss, nsamples);
    reduceSum<<<1, 256, sizeof(float) * 256>>>(d_loss, d_loss, 256);
}

void KmeansGPU::getDistance(const float *d_data)
{
    calDistWithCuda(d_data, d_clusters, d_distances, d_sampleClasses, d_minDist,
                    d_loss, d_clusterSize, m_numClusters, m_nsamples, m_numFeatures);
}

void KmeansGPU::updateClusters(const float *d_data)
{
    updateClusterWithCuda(d_data,
                          d_clusters,
                          d_sampleClasses,
                          d_clusterSize,
                          m_nsamples,
                          m_numClusters,
                          m_numFeatures);
}

void KmeansGPU::fit(const float *v_data)
{
    float *d_data;
    int datamem = sizeof(float) * m_nsamples * m_numFeatures;
    int clustermem = sizeof(float) * m_numClusters * m_numFeatures;
    int sampleClassmem = sizeof(int) * m_nsamples;
    int distmem = sizeof(float) * m_nsamples * m_numClusters;
    int *h_clusterSize = new int[m_numClusters]{0};
    float *h_loss = new float[m_nsamples]{0.0};

    CHECK(cudaMalloc((void **)&d_data, datamem));
    CHECK(cudaMalloc((void **)&d_clusters, clustermem));
    CHECK(cudaMalloc((void **)&d_sampleClasses, sampleClassmem));
    CHECK(cudaMalloc((void **)&d_distances, distmem));
    CHECK(cudaMalloc((void **)&d_minDist, sizeof(float) * m_nsamples));
    CHECK(cudaMalloc((void **)&d_loss, sizeof(float) * m_nsamples));
    CHECK(cudaMalloc((void **)&d_clusterSize, sizeof(int) * m_numClusters));

    CHECK(cudaMemcpy(d_data, v_data, datamem, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_clusters, m_clusters, clustermem, cudaMemcpyHostToDevice));

    float lastLoss = 0;
    for (int i = 0; i < m_maxIters; ++i)
    {
        this->getDistance(d_data);
        this->updateClusters(d_data);
        CHECK(cudaMemcpy(h_loss, d_loss, sampleClassmem, cudaMemcpyDeviceToHost));
        this->m_optTarget = h_loss[0];
        if (std::abs(lastLoss - this->m_optTarget) < this->m_epsilon){
            std::cout << "迭代步长已经小于epsilon!!!" << std:: endl;
            break;
        }
        lastLoss = this->m_optTarget;
        std::cout << "Iters: " << i + 1 << "  current loss : " << m_optTarget << std::endl;
    }

    CHECK(cudaMemcpy(m_clusters, d_clusters, clustermem, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(m_sampleClasses, d_sampleClasses, sampleClassmem, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(m_distances, d_distances, distmem, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_data));
    CHECK(cudaFree(d_clusters));
    CHECK(cudaFree(d_sampleClasses));
    CHECK(cudaFree(d_distances));
    CHECK(cudaFree(d_minDist));
    CHECK(cudaFree(d_loss));
    CHECK(cudaFree(d_clusterSize));
    delete[] h_clusterSize;
    delete[] h_loss;
}

