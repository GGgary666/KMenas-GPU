#include <stdio.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <string>
#include <sstream>
#include <stdexcept>

using namespace std;

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

class Kmeans {
public:
    Kmeans(int numClusters, int numFeatures, float *clusters, int nsamples);
    Kmeans(int numClusters, int numFeatures, float *clusters, int nsamples, 
        int maxIters, float epsilon);
    ~Kmeans();
    virtual void getDistance(const float *v_data); 
    virtual void updateClusters(const float *v_data);
    virtual void fit(const float *v_data);
    virtual float accuracy(const int *v_label);

    float *m_clusters; //[numClusters, numFeatures]
    int m_numClusters;
    int m_numFeatures;
    float *m_distances; // [nsamples, numClusters]
    int *m_sampleClasses; // [nsamples, ]
    int m_nsamples;
    float m_optTarget;
    int m_maxIters;
    float m_epsilon;
private:
    Kmeans(const Kmeans& model);
    Kmeans& operator=(const Kmeans& model);
};

Kmeans::Kmeans(int numClusters, int numFeatures, float *clusters, int nsamples) : m_numClusters(numClusters), m_numFeatures(numFeatures), m_maxIters(50),
                                                                                  m_optTarget(1e7), m_epsilon(0.001), m_nsamples(nsamples)
{
    m_clusters = new float[numClusters * numFeatures];
    for (int i = 0; i < this->m_numClusters * this->m_numFeatures; ++i)
    {
        this->m_clusters[i] = clusters[i];
    }
    m_distances = new float[nsamples * numClusters]{0.0};
    m_sampleClasses = new int[nsamples]{0};
}

Kmeans::Kmeans(int numClusters, int numFeatures, float *clusters, int nsamples,
               int maxIters, float epsilon) : m_numClusters(numClusters), m_numFeatures(numFeatures), m_maxIters(maxIters),
                                              m_optTarget(1e7), m_epsilon(epsilon), m_nsamples(nsamples)
{
    m_clusters = new float[numClusters * numFeatures];
    for (int i = 0; i < this->m_numClusters * this->m_numFeatures; ++i)
    {
        this->m_clusters[i] = clusters[i];
    }
    m_distances = new float[nsamples * numClusters]{0.0};
    m_sampleClasses = new int[nsamples]{0};
}

Kmeans::~Kmeans()
{
    if (m_clusters)
        delete[] m_clusters;
    if (m_distances)
        delete[] m_distances;
    if (m_sampleClasses)
        delete[] m_sampleClasses;
}

void Kmeans::getDistance(const float *v_data)
{
    /*
        v_data: [nsamples, numFeatures, ]
    */

    float loss = 0.0;
    for (int i = 0; i < m_nsamples; ++i)
    {
        float minDist = 1e8;
        int minIdx = -1;
        for (int j = 0; j < m_numClusters; ++j)
        {
            float sum = 0.0;
            for (int k = 0; k < m_numFeatures; ++k)
            {
                sum += (v_data[i * m_numFeatures + k] - m_clusters[j * m_numFeatures + k]) *
                       (v_data[i * m_numFeatures + k] - m_clusters[j * m_numFeatures + k]);
            }
            this->m_distances[i * m_numClusters + j] = sqrt(sum);
            if (sum <= minDist)
            {
                minDist = sum;
                minIdx = j;
            }
        }
        m_sampleClasses[i] = minIdx;
        loss += m_distances[i * m_numClusters + minIdx];
    }
    m_optTarget = loss;
}

void Kmeans::updateClusters(const float *v_data)
{
    for (int i = 0; i < m_numClusters * m_numFeatures; ++i)
        this->m_clusters[i] = 0.0;
    for (int i = 0; i < m_numClusters; ++i)
    {
        int cnt = 0;
        for (int j = 0; j < m_nsamples; ++j)
        {
            if (i != m_sampleClasses[j])
                continue;
            for (int k = 0; k < m_numFeatures; ++k)
            {
                this->m_clusters[i * m_numFeatures + k] += v_data[j * m_numFeatures + k];
            }
            cnt++;
        }
        for (int ii = 0; ii < m_numFeatures; ii++)
            this->m_clusters[i * m_numFeatures + ii] /= cnt;
    }
}

void Kmeans::fit(const float *v_data)
{
    float lastLoss = this->m_optTarget;
    for (int i = 0; i < m_maxIters; ++i)
    {
        this->getDistance(v_data);
        this->updateClusters(v_data);
        if (std::abs(lastLoss - this->m_optTarget) < this->m_epsilon){
            std::cout << "迭代步长已经小于epsilon!!!" << std:: endl;
            break;
        }
            
        lastLoss = this->m_optTarget;
        std::cout << "Iters: " << i + 1 << "  current loss : " << m_optTarget << std::endl;
    }
}

float Kmeans::accuracy(const int *v_label){
    // map clusters to labels
    int* mappedLabels = new int[this->m_nsamples];
    
    for(int clusterNum = 0; clusterNum < this->m_numClusters; clusterNum++){
        std::vector<int> clusterIndices;
        // 找到属于当前簇的数据点的索引
        for (size_t i = 0; i < this->m_nsamples; ++i) {
            if (this->m_sampleClasses[i] == clusterNum) {
                clusterIndices.push_back(i);
            }
        }   
        // 统计当前簇中真实标签出现的频率
        std::unordered_map<int, int> labelFreq;
        for (int index : clusterIndices) {
            ++labelFreq[v_label[index]];
        }

        // 找到当前簇中出现频率最高的真实标签
        int mostFrequentLabel = -1;
        int maxFreq = 0;
        for (const auto& pair : labelFreq) {
            if (pair.second > maxFreq) {
                mostFrequentLabel = pair.first;
                maxFreq = pair.second;
            }
        }

        // 将当前簇中的标签映射为出现频率最高的真实标签
        for (int index : clusterIndices) {
            mappedLabels[index] = mostFrequentLabel;
        }
    }

    int count = 0;
    for(int i = 0; i < this->m_nsamples; i++){
        // if(i < 100){
        //     std::cout << "--------------------------------------------------------" << std:: endl;
        //     std::cout << "m_sampleClasses[i]: " << m_sampleClasses[i] << std::endl;
        //     std::cout << "v_label[i]: " << v_label[i] << std::endl;
        // }
        
        if(v_label[i] == mappedLabels[i]) count++;
    }
    float res = float(count) / float(this->m_nsamples);
    return res;
}

class KmeansGPU : public Kmeans
{
public:
    KmeansGPU(int numClusters, int numFeatures, float *clusters, int nsamples);
    KmeansGPU(int numClusters, int numFeatures, float *clusters, int nsamples,
              int maxIters, float eplison);
    ~KmeansGPU();
    virtual void getDistance(const float *d_data);
    virtual void updateClusters(const float *d_data);
    virtual void fit(const float *v_data);
    

    float *d_clusters; // [numClusters, numFeatures]
    int *d_sampleClasses;
    float *d_distances;
    float *d_minDist;   // [nsamples, ]
    float *d_loss;      // [nsamples, ]
    int *d_clusterSize; //[numClusters, ]

private:
    KmeansGPU(const Kmeans &model);
    KmeansGPU &operator=(const Kmeans &model);
};
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
        // if (std::abs(lastLoss - this->m_optTarget) < this->m_epsilon){
        //     std::cout << "迭代步长已经小于epsilon!!!" << std:: endl;
        //     break;
        // }
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


void readCoordinate(float *data, int *label, const int n_features, int &n, string file) {
    std::ifstream ifs;
    ifs.open(file, std::ios::in);
    if (ifs.fail()) {
        std::cout << "No such file or directory: "<< file << std::endl;
        exit(1);
    }
    std::string line;
    while (std::getline(ifs, line)) {
        std::stringstream sstream(line);
        if (line.empty()) continue;
        int m = 0;
        std::string s_fea;
        while (std::getline(sstream, s_fea, ',')) {
            if (m < n_features) data[n * n_features + m] = std::stod(s_fea);
            else label[n] = std::stoi(s_fea);
            m++;
        }
        n++;
    }
    ifs.close();
}

template <typename T>
void printVecInVec(const T* vecInVec, int rows, int cols, const std::string& title) {
    std::cout << title << ":" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << vecInVec[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

void timing(
    float *data, 
    int *label, 
    float *clusters, 
    const int numClusters, 
    const int n_features, 
    const int n_samples,
    const int method) {
    
    Kmeans *model;
    switch (method)
    {
    case 0:
        model = new Kmeans(numClusters, n_features, clusters, n_samples, 50, 1e-5);
        break;
    case 1:
        model = new KmeansGPU(numClusters, n_features, clusters, n_samples, 50, 1e-5);
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

    // std::cout << "********* final clusters**********" << std::endl;
    // printVecInVec<float>(model->m_clusters, 4, 4, "clusters");
    // std::cout << "*********    accuracy  **********" << std::endl;
    // std::cout << "model accuracy : " << model->accuracy(label) << std::endl;
    // printVecInVec<int>(model->m_sampleClasses, 1, 10, "sampleClasses_10");

    delete model;
}


int main() {
    string file = "/home/gg/Desktop/kmeans/data/test_1e8.csv";
    int N = 0; // 实际读取的样本数量
    const int n_nums = 100000000; // 数据中，有100个样本
    const int n_features = 4; // 每个样本有4个特征
    const int n_classes = 4;
    float *data = new float[n_features * n_nums];
    int *label = new int[n_nums];
    readCoordinate(data, label, n_features, N, file);
    std::cout << "num of samples : " << N << std::endl;

    // 数据初始化
    
    int cidx[4] = {0};
    srand(time(NULL));
    for(int i = 0; i < n_classes; i++) cidx[i] = rand() % 100;
    float clusters[n_classes * n_features] = {0};
    for(int i = 0; i < n_classes; i++){
        for(int j = 0; j < n_features; j++){
            clusters[i * n_features + j] = data[cidx[i] * n_features + j];
        }
    }

    // printVecInVec<float>(clusters, 4, 4, "clusters");

    // std::cout << "Using CPU:" << std::endl;
    // timing(data, label, clusters, n_classes, n_features, N, 0);

    std::cout << "Using CUDA:" << std::endl;
    timing(data, label, clusters, n_classes, n_features, N, 1);
    delete[] data;
    delete[] label;
    return 0;
}