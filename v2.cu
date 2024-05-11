#include <stdio.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <string>
#include <sstream>
#include <cstdlib> // for rand() and srand()
#include <ctime>   // for time()
#include <stdexcept>
using namespace std;

#define CHECK(call)                                     \
    do                                                  \
    {                                                   \
        const cudaError_t error_code = call;            \
        if (error_code != cudaSuccess)                  \
        {                                               \
            printf("CUDA Error:\n");                    \
            printf("    File:       %s\n", __FILE__);   \
            printf("    Line:       %d\n", __LINE__);   \
            printf("    Error code: %d\n", error_code); \
            printf("    Error text: %s\n",              \
                   cudaGetErrorString(error_code));     \
            exit(1);                                    \
        }                                               \
    } while (0)

template <typename DataType>
class Kmeans
{
public:
    Kmeans(int num_clusters, int num_features, DataType *clusters, int num_samples);
    Kmeans(int num_clusters, int num_features, DataType *clusters, int num_samples, int max_iters, float eplison);
    virtual ~Kmeans();
    void getDistance(const DataType *v_data);
    void updateClusters(const DataType *v_data);
    virtual void fit(const DataType *v_data);
    float accuracy(int *v_label);

    DataType *m_clusters; //[num_clusters, num_features]
    int m_num_clusters;
    int m_num_features;
    float *m_distances;   // [num_samples, num_clusters]
    int *m_sampleClasses; // [num_samples, ]
    int m_num_samples;
    float m_optTarget;
    int m_max_iters;
    float m_epsilon;

private:
    Kmeans(const Kmeans &model);
    Kmeans &operator=(const Kmeans &model);
};

template <typename DataType>
Kmeans<DataType>::Kmeans(int numClusters, int numFeatures, DataType *clusters, int nsamples) : m_num_clusters(numClusters), m_num_features(numFeatures), m_max_iters(50),
                                                                                               m_optTarget(1e7), m_epsilon(0.001), m_num_samples(nsamples)
{
    m_clusters = new float[numClusters * numFeatures];
    for (int i = 0; i < this->m_num_clusters * this->m_num_features; ++i)
    {
        this->m_clusters[i] = clusters[i];
    }
    m_distances = new float[nsamples * numClusters]{0.0};
    m_sampleClasses = new int[nsamples]{0};
}

template <typename DataType>
Kmeans<DataType>::Kmeans(int numClusters, int numFeatures, DataType *clusters, int nsamples,
                         int maxIters, float epsilon) : m_num_clusters(numClusters), m_num_features(numFeatures), m_max_iters(maxIters),
                                                        m_optTarget(1e7), m_epsilon(epsilon), m_num_samples(nsamples)
{
    m_clusters = new float[numClusters * numFeatures];
    for (int i = 0; i < this->m_num_clusters * this->m_num_features; ++i)
    {
        this->m_clusters[i] = clusters[i];
    }
    m_distances = new float[nsamples * numClusters]{0.0};
    m_sampleClasses = new int[nsamples]{0};
}

template <typename DataType>
Kmeans<DataType>::~Kmeans()
{
    if (m_clusters)
        delete[] m_clusters;
    if (m_distances)
        delete[] m_distances;
    if (m_sampleClasses)
        delete[] m_sampleClasses;
}

template <typename DataType>
void Kmeans<DataType>::getDistance(const DataType *v_data)
{
    /*
        v_data: [nsamples, numFeatures, ]
    */

    float loss = 0.0;
    for (int i = 0; i < m_num_samples; ++i)
    {
        float minDist = 1e8;
        int minIdx = -1;
        for (int j = 0; j < m_num_clusters; ++j)
        {
            float sum = 0.0;
            for (int k = 0; k < m_num_features; ++k)
            {
                sum += (v_data[i * m_num_features + k] - m_clusters[j * m_num_features + k]) *
                       (v_data[i * m_num_features + k] - m_clusters[j * m_num_features + k]);
            }
            this->m_distances[i * m_num_clusters + j] = sqrt(sum);
            if (sum <= minDist)
            {
                minDist = sum;
                minIdx = j;
            }
        }
        m_sampleClasses[i] = minIdx;
        loss += m_distances[i * m_num_clusters + minIdx];
    }
    m_optTarget = loss;
}

template <typename DataType>
void Kmeans<DataType>::updateClusters(const DataType *v_data)
{
    for (int i = 0; i < m_num_clusters * m_num_features; ++i)
        this->m_clusters[i] = 0.0;
    for (int i = 0; i < m_num_clusters; ++i)
    {
        int cnt = 0;
        for (int j = 0; j < m_num_samples; ++j)
        {
            if (i != m_sampleClasses[j])
                continue;
            for (int k = 0; k < m_num_features; ++k)
            {
                this->m_clusters[i * m_num_features + k] += v_data[j * m_num_features + k];
            }
            cnt++;
        }
        for (int ii = 0; ii < m_num_features; ii++)
            this->m_clusters[i * m_num_features + ii] /= cnt;
    }
}

template <typename DataType>
void Kmeans<DataType>::fit(const DataType *v_data)
{
    float lastLoss = this->m_optTarget;
    for (int i = 0; i < m_max_iters; ++i)
    {
        this->getDistance(v_data);
        this->updateClusters(v_data);
        if (std::abs(lastLoss - this->m_optTarget) < this->m_epsilon)
        {
            std::cout << "迭代步长已经小于epsilon!!!" << std::endl;
            break;
        }

        lastLoss = this->m_optTarget;
        std::cout << "Iters: " << i + 1 << "  current loss : " << m_optTarget << std::endl;
    }
}

template <typename DataType>
float Kmeans<DataType>::accuracy(int *v_label)
{
    // map clusters to labels
    int *mappedLabels = new int[this->m_num_samples];

    for (int clusterNum = 0; clusterNum < this->m_num_clusters; clusterNum++)
    {
        std::vector<int> clusterIndices;
        // 找到属于当前簇的数据点的索引
        for (size_t i = 0; i < this->m_num_samples; ++i)
        {
            if (this->m_sampleClasses[i] == clusterNum)
            {
                clusterIndices.push_back(i);
            }
        }
        // 统计当前簇中真实标签出现的频率
        std::unordered_map<int, int> labelFreq;
        for (int index : clusterIndices)
        {
            ++labelFreq[v_label[index]];
        }

        // 找到当前簇中出现频率最高的真实标签
        int mostFrequentLabel = -1;
        int maxFreq = 0;
        for (const auto &pair : labelFreq)
        {
            if (pair.second > maxFreq)
            {
                mostFrequentLabel = pair.first;
                maxFreq = pair.second;
            }
        }

        // 将当前簇中的标签映射为出现频率最高的真实标签
        for (int index : clusterIndices)
        {
            mappedLabels[index] = mostFrequentLabel;
        }
    }

    int count = 0;
    for (int i = 0; i < this->m_num_samples; i++)
    {
        // if(i < 100){
        //     std::cout << "--------------------------------------------------------" << std:: endl;
        //     std::cout << "m_sampleClasses[i]: " << m_sampleClasses[i] << std::endl;
        //     std::cout << "v_label[i]: " << v_label[i] << std::endl;
        // }

        if (v_label[i] == mappedLabels[i])
            count++;
    }
    float res = float(count) / float(this->m_num_samples);
    return res;
}

template <typename DataType>
class KmeansGPUV2 : public Kmeans<DataType>
{
public:
    KmeansGPUV2(int num_clusters, int num_features, DataType *clusters, int num_samples);
    KmeansGPUV2(int num_clusters, int num_features, DataType *clusters, int num_samples,
                int max_iters, float eplison);
    ~KmeansGPUV2();
    void getDistance(const DataType *v_data);
    void updateClusters(const DataType *v_data);
    void fit(const DataType *v_data);

    DataType *d_data;     // [num_samples, num_features]
    DataType *d_clusters; // [num_clusters, num_features]
    int *d_sampleClasses; // [num_samples, ]
    float *d_min_dist;    // [num_samples, ]
    float *d_loss;        // [num_samples, ]
    int *d_cluster_size;  //[num_clusters, ]
    cudaStream_t master_stream;

private:
    KmeansGPUV2(const Kmeans<DataType> &model);
    KmeansGPUV2 &operator=(const Kmeans<DataType> &model);
};

template <typename DataType>
KmeansGPUV2<DataType>::KmeansGPUV2(int num_clusters, int num_features, DataType *clusters, int num_samples,
                                   int max_iters, float eplison)
    : Kmeans<DataType>(num_clusters, num_features, clusters, num_samples, max_iters, eplison)
{
    CHECK(cudaStreamCreate(&master_stream));

    int data_buf_size = this->m_num_samples * this->m_num_features;
    int cluster_buf_size = this->m_num_clusters * this->m_num_features;
    int mem_size = sizeof(DataType) * (data_buf_size + cluster_buf_size) + sizeof(int) * (this->m_num_samples) +
                   sizeof(float) * (this->m_num_samples + this->m_num_samples) + sizeof(int) * this->m_num_clusters;

    CHECK(cudaMalloc((void **)&d_data, mem_size));

    d_clusters = (DataType *)(d_data + data_buf_size);
    d_sampleClasses = (int *)(d_clusters + cluster_buf_size);
    d_min_dist = (float *)(d_sampleClasses + this->m_num_samples);
    d_loss = (float *)(d_min_dist + this->m_num_samples);
    d_cluster_size = (int *)(d_loss + this->m_num_samples);

    CHECK(cudaMemcpy(d_clusters, this->m_clusters, sizeof(DataType) * cluster_buf_size, cudaMemcpyHostToDevice));
}

template <typename DataType>
KmeansGPUV2<DataType>::~KmeansGPUV2()
{
    CHECK(cudaFree(d_data));
    CHECK(cudaStreamDestroy(master_stream));
}

template <typename DataType>
__global__ void initV2(DataType *x, const DataType value, const int N)
{
    const int n = threadIdx.x + blockIdx.x * blockDim.x;
    if (n < N)
        x[n] = value;
}

template <typename T>
struct SumOp
{
    __device__ __forceinline__ T operator()(const T &a, const T &b) const { return a + b; }
};

template <template <typename> class ReductionOp, typename T>
__inline__ __device__ T WarpReduce(T val)
{
    auto func = ReductionOp<T>();
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
    {
        val = func(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
    }
    return val;
}

template <template <typename> class ReductionOp, typename T>
__inline__ __device__
    T
    blockReduce(T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f;
    int wid = threadIdx.x >> 5;

    val = WarpReduce<ReductionOp, T>(val);

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)0.0f;
    val = WarpReduce<ReductionOp, T>(val);
    return val;
}

template <template <typename> class ReductionOp>
__global__ void vec1DReduce(float *vec, float *reduce, const int N)
{
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    float val = 0.0f;

    auto func = ReductionOp<float>();

#pragma unroll
    for (; n < N; n += blockDim.x * gridDim.x)
        val = func(val, vec[n]);
    __syncthreads();

    float block_sum = blockReduce<ReductionOp, float>(val);
    if (threadIdx.x == 0)
        reduce[blockIdx.x] = block_sum;
}

template <typename DataType>
__device__ float calDistV2(const DataType *d_data,
                           const DataType *d_clusters, // [num_clusters, num_features]
                           const int clusterNo, const int num_features)
{
    // grid_size = num_samples, block_size = 256
    const int sample_offset = num_features * blockIdx.x;
    const int cluster_offset = num_features * clusterNo;

    float distance = 0.0f;
    float sub_val;

#pragma unroll
    for (int i = threadIdx.x; i < num_features; i += blockDim.x)
    {
        sub_val = (float)(d_data[sample_offset + i] - d_clusters[cluster_offset + i]);
        distance += sub_val * sub_val;
    }
    __syncthreads();

    distance = blockReduce<SumOp, float>(distance);
    return distance;
}

template <typename DataType>
__global__ void calClustersDistkernel(const DataType *d_data,
                                      const DataType *d_clusters, // [num_clusters, num_features]
                                      int *d_sample_classes,      // [nsamples, ]
                                      float *d_min_dist,          // [nsamples, ]
                                      const int num_features,
                                      const int num_clusters)
{
    // grid_size = num_samples, block_size = 128
    float min_dist = 1e9f;
    float dist;
    int min_idx;

#pragma unroll
    for (int i = 0; i < num_clusters; ++i)
    {
        dist = calDistV2<DataType>(d_data, d_clusters, i, num_features);
        if (dist < min_dist)
        {
            min_dist = dist;
            min_idx = i;
        }
    }

    if (threadIdx.x == 0)
    {
        d_sample_classes[blockIdx.x] = min_idx;
        d_min_dist[blockIdx.x] = sqrtf(min_dist);
    }
}

__global__ void histCount(int *d_sample_classes, // [N, ]
                          int *d_clusterSize,    // [num_clusters, ]
                          const int num_clusters, const int N)
{
    // block_size = 128, grid_size = (num_samples - 1) / block_size + 1;
    int n = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ int s_histo[256];
    if (threadIdx.x < num_clusters)
        s_histo[threadIdx.x] = 0;
    __syncthreads();

#pragma unroll
    for (; n < N; n += gridDim.x * blockDim.x)
    {
        atomicAdd(&s_histo[d_sample_classes[n]], 1);
    }
    __syncthreads();
    if (threadIdx.x < num_clusters)
        atomicAdd(&d_clusterSize[threadIdx.x], s_histo[threadIdx.x]);
}

template <typename DataType>
void calDistWithCudaV2(const DataType *d_data, DataType *d_clusters, int *d_sample_classes, int *d_cluster_size,
                       float *d_min_dist, float *d_loss, const int num_clusters, const int num_samples, const int num_features,
                       cudaStream_t master_stream)
{
    cudaEvent_t cal_dist_event;
    CHECK(cudaEventCreate(&cal_dist_event));
    const int block_size = 128;
    calClustersDistkernel<DataType><<<num_samples, block_size, 0, master_stream>>>(
        d_data, d_clusters, d_sample_classes, d_min_dist, num_features, num_clusters);

    CHECK(cudaEventRecord(cal_dist_event, master_stream));

    cudaStream_t tmp_stream[2];
    CHECK(cudaStreamCreate(&tmp_stream[0]));
    CHECK(cudaStreamCreate(&tmp_stream[1]));

    CHECK(cudaStreamWaitEvent(tmp_stream[0], cal_dist_event));
    CHECK(cudaStreamWaitEvent(tmp_stream[1], cal_dist_event));

    vec1DReduce<SumOp><<<block_size, block_size, 0, master_stream>>>(d_min_dist, d_loss, num_samples);
    vec1DReduce<SumOp><<<1, block_size, 0, master_stream>>>(d_loss, d_loss, block_size);

    initV2<int><<<1, 1024, 0, tmp_stream[0]>>>(d_cluster_size, 0.0f, num_clusters);
    int grid_size = (num_samples - 1) / block_size + 1;
    histCount<<<grid_size, block_size, 0, tmp_stream[0]>>>(d_sample_classes, d_cluster_size, num_clusters, num_samples);

    initV2<DataType><<<1, 1024, 0, tmp_stream[1]>>>(d_clusters, 0.0f, num_clusters * num_features);

    CHECK(cudaStreamDestroy(tmp_stream[1]));
    CHECK(cudaStreamDestroy(tmp_stream[0]));

    CHECK(cudaEventDestroy(cal_dist_event));
}

template <typename DataType>
void KmeansGPUV2<DataType>::getDistance(const DataType *d_data)
{
    calDistWithCudaV2<float>(d_data, d_clusters, d_sampleClasses, d_cluster_size, d_min_dist, d_loss, this->m_num_clusters, this->m_num_samples, this->m_num_features, master_stream);
}

template <typename DataType>
__global__ void update(
    const DataType *d_data,
    DataType *d_clusters,
    int *d_sampleClasses,
    int *d_cluster_size,
    const int num_samples,
    const int num_features)
{
    // grid_size = num_samples, block_size = block_size
    int clusterId = d_sampleClasses[blockIdx.x];
    int clustercnt = d_cluster_size[clusterId];

#pragma unroll
    for (int i = threadIdx.x; i < num_features; i += blockDim.x)
    {
        atomicAdd(&(d_clusters[clusterId * num_features + i]), d_data[num_features * blockIdx.x + i] / clustercnt);
    }
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

template <typename DataType>
void updateClusterWithCuda(
    const DataType *d_data,
    DataType *d_clusters,
    int *d_sampleClasses,
    int *d_clusterSize,
    const int nsamples,
    const int numClusters,
    const int numFeatures)
{

    initV2<DataType><<<1, 1024>>>(d_clusters, 0.0, numClusters * numFeatures);
    int blockSize = 1024;
    int gridSize = (nsamples - 1) / blockSize + 1;
    countCluster<<<gridSize, blockSize>>>(d_sampleClasses, d_clusterSize, nsamples);
    update<DataType><<<nsamples, 128>>>(d_data, d_clusters, d_sampleClasses, d_clusterSize, nsamples, numFeatures);
}

template <typename DataType>
void KmeansGPUV2<DataType>::updateClusters(const DataType *d_data)
{
    updateClusterWithCuda<DataType>(d_data,
                                    d_clusters,
                                    d_sampleClasses,
                                    d_cluster_size,
                                    this->m_num_samples,
                                    this->m_num_clusters,
                                    this->m_num_features);
}

template <typename DataType>
void KmeansGPUV2<DataType>::fit(const DataType *v_data)
{
    DataType *d_data;
    int datamem = sizeof(DataType) * this->m_num_samples * this->m_num_features;
    int clustermem = sizeof(DataType) * this->m_num_clusters * this->m_num_features;
    int sampleClassmem = sizeof(int) * this->m_num_samples;
    int distmem = sizeof(float) * this->m_num_samples * this->m_num_clusters;
    int *h_clusterSize = new int[this->m_num_clusters]{0};
    float *h_loss = new float[this->m_num_samples]{0.0};

    CHECK(cudaMalloc((void **)&d_data, datamem));
    CHECK(cudaMalloc((void **)&d_clusters, clustermem));
    CHECK(cudaMalloc((void **)&d_sampleClasses, sampleClassmem));
    CHECK(cudaMalloc((void **)&d_min_dist, sizeof(float) * this->m_num_samples));
    CHECK(cudaMalloc((void **)&d_loss, sizeof(float) * this->m_num_samples));
    CHECK(cudaMalloc((void **)&d_cluster_size, sizeof(int) * this->m_num_clusters));

    CHECK(cudaMemcpy(d_data, v_data, datamem, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_clusters, this->m_clusters, clustermem, cudaMemcpyHostToDevice));

    float lastLoss = 0;
    for (int i = 0; i < this->m_max_iters; ++i)
    {
        this->getDistance(d_data);
        this->updateClusters(d_data);
        CHECK(cudaMemcpy(h_loss, d_loss, sampleClassmem, cudaMemcpyDeviceToHost));
        this->m_optTarget = h_loss[0];
        // if (std::abs(lastLoss - this->m_optTarget) < this->m_epsilon)
        // {
        //     std::cout << "迭代步长已经小于epsilon!!!" << std::endl;
        //     break;
        // }
        lastLoss = this->m_optTarget;
        std::cout << "Iters: " << i + 1 << "  current loss : " << this->m_optTarget << std::endl;
    }

    CHECK(cudaMemcpy(this->m_clusters, d_clusters, clustermem, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(this->m_sampleClasses, d_sampleClasses, sampleClassmem, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_data));
    CHECK(cudaFree(d_clusters));
    CHECK(cudaFree(d_sampleClasses));
    CHECK(cudaFree(d_min_dist));
    CHECK(cudaFree(d_loss));
    CHECK(cudaFree(d_cluster_size));
    delete[] h_clusterSize;
    delete[] h_loss;
}



void readCoordinate(float *data, int *label, const int n_features, int &n, string file)
{
    std::ifstream ifs;
    ifs.open(file, std::ios::in);
    if (ifs.fail())
    {
        std::cout << "No such file or directory: "<< file << std::endl;
        exit(1);
    }
    std::string line;
    while (std::getline(ifs, line))
    {
        std::stringstream sstream(line);
        if (line.empty())
            continue;
        int m = 0;
        std::string s_fea;
        while (std::getline(sstream, s_fea, ','))
        {
            if (m < n_features)
                data[n * n_features + m] = std::stod(s_fea);
            else
                label[n] = std::stoi(s_fea);
            m++;
        }
        n++;
    }
    ifs.close();
}

template <typename T>
void printVecInVec(const T *vecInVec, int rows, int cols, const std::string &title)
{
    std::cout << title << ":" << std::endl;
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            std::cout << vecInVec[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}



template <typename DataType>
void timing(
    DataType *data,
    int *label,
    DataType *clusters,
    const int numClusters,
    const int n_features,
    const int n_samples,
    const int method)
{

    Kmeans<DataType> *model;
    switch (method)
    {
    case 0:
        model = new Kmeans<DataType>(numClusters, n_features, clusters, n_samples, 50, 1e-5);
        break;
    case 1:
        model = new KmeansGPUV2<DataType>(numClusters, n_features, clusters, n_samples, 50, 1e-5);
        break;
    default:
        break;
    }

    std::cout << "*********starting fitting*********" << std::endl;

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    cudaEventQuery(start);

    model->fit(data);

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float elapsedTime;
    CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    printf("Time = %g ms.\n", elapsedTime);

    std::cout << "********* final clusters**********" << std::endl;
    printVecInVec<float>(model->m_clusters, 4, n_features, "clusters");
    std::cout << "*********    accuracy  **********" << std::endl;
    std::cout << "model accuracy : " << model->accuracy(label) << std::endl;
    printVecInVec<int>(model->m_sampleClasses, 1, 10, "sampleClasses_10");

    delete model;
}

int main()
{   
    string file = "/home/gg/Desktop/kmeans/data/test_1e7.csv";
    int N = 0;                // 实际读取的样本数量
    const int n_nums = 10000000;   // 数据中，有100个样本
    const int n_features = 4; // 每个样本有4个特征
    const int n_classes = 4;
    float *data = new float[n_features * n_nums];
    int *label = new int[n_nums];
    readCoordinate(data, label, n_features, N, file);
    std::cout << "num of samples : " << N << std::endl;

    // 数据初始化

    int cidx[4] = {0};
    srand(time(NULL));
    for (int i = 0; i < n_classes; i++)
        cidx[i] = rand() % 100;
    float clusters[n_classes * n_features] = {0};
    for (int i = 0; i < n_classes; i++)
    {
        for (int j = 0; j < n_features; j++)
        {
            clusters[i * n_features + j] = data[cidx[i] * n_features + j];
        }
    }

    printVecInVec<float>(clusters, 4, 4, "clusters");

    // std::cout << "Using CPU:" << std::endl;
    // timing<float>(data, label, clusters, n_classes, n_features, N, 0);

    std::cout << "Using CUDA:" << std::endl;
    timing<float>(data, label, clusters, n_classes, n_features, N, 1);
    delete[] data;
    delete[] label;
    return 0;
}

