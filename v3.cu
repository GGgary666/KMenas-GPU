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
class KmeansGPUV3 : public Kmeans<DataType>
{
public:
    KmeansGPUV3(int num_clusters, int num_features, DataType *clusters, int num_samples);
    KmeansGPUV3(int num_clusters, int num_features, DataType *clusters, int num_samples,
                int max_iters, float eplison);
    virtual ~KmeansGPUV3();
    void fit(const DataType *v_data);

    DataType *d_data;     // [num_samples, num_features]
    DataType *d_clusters; // [num_clusters, num_features]
    int *d_sampleClasses; // [num_samples, ]
    float *d_min_dist;    // [num_samples, ]
    float *d_loss;        // [num_samples, ]
    int *d_cluster_size;  //[num_clusters, ]
    cudaStream_t calculate_stream;
    cudaStream_t update_stream;
    cudaEvent_t calculate_event;
    cudaEvent_t update_event;

private:
    KmeansGPUV3(const Kmeans<DataType> &model);
    KmeansGPUV3 &operator=(const Kmeans<DataType> &model);
};

template <typename DataType>
KmeansGPUV3<DataType>::KmeansGPUV3(int num_clusters, int num_features, DataType *clusters, int num_samples,
                                   int max_iters, float eplison)
    : Kmeans<DataType>(num_clusters, num_features, clusters, num_samples, max_iters, eplison)
{
    CHECK(cudaStreamCreate(&calculate_stream));
    CHECK(cudaStreamCreate(&update_stream));
    CHECK(cudaEventCreate(&calculate_event));
    CHECK(cudaEventCreate(&update_event));

    int data_buf_size = this->m_num_samples * this->m_num_features;
    int cluster_buf_size = this->m_num_clusters * this->m_num_features;
    int mem_size = sizeof(DataType) * (data_buf_size + cluster_buf_size) + sizeof(int) * (this->m_num_samples) +
                   sizeof(float) * (this->m_num_samples + this->m_num_samples) + sizeof(int) * this->m_num_clusters;

    CHECK(cudaMallocAsync((void **)&d_data, mem_size, calculate_stream));

    d_clusters = (DataType *)(d_data + data_buf_size);
    d_sampleClasses = (int *)(d_clusters + cluster_buf_size);
    d_min_dist = (float *)(d_sampleClasses + this->m_num_samples);
    d_loss = (float *)(d_min_dist + this->m_num_samples);
    d_cluster_size = (int *)(d_loss + this->m_num_samples);

    CHECK(cudaMemcpyAsync(d_clusters, this->m_clusters, sizeof(DataType) * cluster_buf_size, cudaMemcpyHostToDevice, update_stream));
    CHECK(cudaEventRecord(update_event, update_stream));

    printf("num_samples: %d  num_clusters: %d  num_features: %d\n", num_samples, num_clusters, num_features);
}

template <typename DataType>
KmeansGPUV3<DataType>::~KmeansGPUV3()
{
    CHECK(cudaFreeAsync(d_data, calculate_stream));
    CHECK(cudaStreamDestroy(calculate_stream));
    CHECK(cudaStreamDestroy(update_stream));
    CHECK(cudaEventDestroy(calculate_event));
    CHECK(cudaEventDestroy(update_event));
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

constexpr int kMaxPackBytes = 128 / 8; //  CUDA 最多支持 128 个 bit 的访问粒度
constexpr int kMaxPackSize = 8;        // half 类型占 2 个字节，也就是 16 个 bit，所以最大可以 Pack 的数量为 128 / 16 = 8

constexpr int Min(int a, int b)
{
    return a < b ? a : b;
}

template <typename T>
constexpr int PackSize()
{
    return Min(kMaxPackBytes / sizeof(T), kMaxPackSize);
}

template <typename T, typename U, typename... Args>
constexpr int PackSize()
{
    return Min(PackSize<T>(), PackSize<U, Args...>());
}

template <typename T, int N>
struct GetPackType
{
    using type = typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;
};

template <typename T, int N>
using PackType = typename GetPackType<T, N>::type;

template <typename T, int pack_size>
struct alignas(sizeof(T) * pack_size) Packed
{
    __device__ Packed()
    {
        // do nothing
    }
    union
    {
        PackType<T, pack_size> storage;
        T elem[pack_size];
    };
};

template <typename DataType, int pack_size>
__device__ float calDistPacked(const DataType *d_data,
                               const DataType *d_clusters, // [num_clusters, num_features]
                               const int clusterNo, const int num_features)
{
    // grid_size = num_samples, block_size = 128
    const int sample_offset = num_features * blockIdx.x;
    const int cluster_offset = num_features * clusterNo;

    const PackType<DataType, pack_size> *buf = reinterpret_cast<const PackType<DataType, pack_size> *>(d_data + sample_offset);
    const PackType<DataType, pack_size> *cluster_buf = reinterpret_cast<const PackType<DataType, pack_size> *>(d_clusters + cluster_offset);
    int num_packs = num_features / pack_size;

    float distance = 0.0f;
    float sub_val;
    Packed<DataType, pack_size> data_pack;
    Packed<DataType, pack_size> cluster_pack;

#pragma unroll
    for (int pack_id = threadIdx.x; pack_id < num_packs; pack_id += blockDim.x)
    {
        data_pack.storage = *(buf + pack_id);
        cluster_pack.storage = *(cluster_buf + pack_id);
#pragma unroll
        for (int elem_id = 0; elem_id < pack_size; ++elem_id)
        {
            sub_val = (float)(data_pack.elem[elem_id] - cluster_pack.elem[elem_id]);
            distance += sub_val * sub_val;
        }
    }
    __syncthreads();

    distance = blockReduce<SumOp, float>(distance);
    return distance;
}

template <typename DataType, int pack_size>
__global__ void calClustersDistPackedkernel(const DataType *d_data,
                                            const DataType *d_clusters, // [num_clusters, num_features]
                                            int *d_sample_classes,      // [nsamples, ]
                                            float *d_min_dist,          // [nsamples, ]
                                            int *d_clusterSize,         // [nsamples, ]
                                            const int num_features,
                                            const int num_clusters)
{
    // grid_size = num_samples, block_size = 256
    float min_dist = 1e9f;
    float dist;
    int min_idx;

#pragma unroll
    for (int i = 0; i < num_clusters; ++i)
    {
        dist = calDistPacked<DataType, pack_size>(d_data, d_clusters, i, num_features);
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
        atomicAdd(&(d_clusterSize[min_idx]), 1);
    }
}

template <typename DataType>
void launchFit(const DataType *d_data, DataType *d_clusters, int *d_sample_classes,
               int *d_cluster_size, float *d_min_dist, float *d_loss, const int num_clusters,
               const int num_samples, const int num_features, cudaStream_t calculate_stream,
               cudaStream_t update_stream, cudaEvent_t calculate_event, cudaEvent_t update_event)
{
    CHECK(cudaStreamWaitEvent(calculate_stream, update_event));

    initV2<int><<<1, 1024, 0, calculate_stream>>>(d_cluster_size, 0.0f, num_clusters);
    const int block_size = 32;
    if (num_features % 4)
    {
        calClustersDistPackedkernel<DataType, 1><<<num_samples, block_size, 0, calculate_stream>>>(d_data, d_clusters,
                                                                                                   d_sample_classes, d_min_dist, d_cluster_size, num_features, num_clusters);
    }
    else
    {
        calClustersDistPackedkernel<DataType, 4><<<num_samples, block_size, 0, calculate_stream>>>(d_data, d_clusters,
                                                                                                   d_sample_classes, d_min_dist, d_cluster_size, num_features, num_clusters);
    }
    CHECK(cudaEventRecord(calculate_event, calculate_stream));

    vec1DReduce<SumOp><<<block_size, block_size, 0, calculate_stream>>>(d_min_dist, d_loss, num_samples);
    vec1DReduce<SumOp><<<1, block_size, 0, calculate_stream>>>(d_loss, d_loss, block_size);

    CHECK(cudaStreamWaitEvent(update_stream, calculate_event));

    initV2<DataType><<<1, 1024, 0, update_stream>>>(d_clusters, 0.0f, num_clusters * num_features);
    update<DataType><<<num_samples, block_size, 0, update_stream>>>(d_data, d_clusters,
                                                                    d_sample_classes, d_cluster_size, num_samples, num_features);
    CHECK(cudaEventRecord(update_event, update_stream));
}

template <typename DataType>
void KmeansGPUV3<DataType>::fit(const DataType *v_data)
{
    float *h_loss = new float[this->m_num_samples]{0.0};
    CHECK(cudaMemcpyAsync(d_data, v_data, sizeof(DataType) * this->m_num_samples * this->m_num_features, cudaMemcpyHostToDevice, calculate_stream));

    float lastLoss = 0;
    for (int i = 0; i < this->m_max_iters; ++i)
    {
        launchFit<DataType>(d_data, d_clusters, d_sampleClasses, d_cluster_size, d_min_dist, d_loss,
                            this->m_num_clusters, this->m_num_samples, this->m_num_features, calculate_stream, update_stream,
                            calculate_event, update_event);

        CHECK(cudaMemcpyAsync(h_loss, d_loss, sizeof(float) * this->m_num_samples, cudaMemcpyDeviceToHost, calculate_stream));
        CHECK(cudaStreamSynchronize(calculate_stream));
        this->m_optTarget = h_loss[0];
        // if (std::abs(lastLoss - this->m_optTarget) < this->m_epsilon)
        // {
        //     std::cout << "迭代步长已经小于epsilon!!!" << std::endl;
        //     break;
        // }
        lastLoss = this->m_optTarget;
        std::cout << "Iters: " << i + 1 << "  current loss : " << this->m_optTarget << std::endl;
    }

    CHECK(cudaMemcpyAsync(this->m_clusters, d_clusters, sizeof(DataType) * this->m_num_clusters * this->m_num_features, cudaMemcpyDeviceToHost, calculate_stream));
    CHECK(cudaMemcpyAsync(this->m_sampleClasses, d_sampleClasses, sizeof(int) * this->m_num_samples, cudaMemcpyDeviceToHost, calculate_stream));

    delete[] h_loss;
}

void readCoordinate(float *data, int *label, const int n_features, int &n, string file)
{
    std::ifstream ifs;
    ifs.open(file, std::ios::in);
    if (ifs.fail())
    {
        std::cout << "No such file or directory: " << file << std::endl;
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
        model = new KmeansGPUV3<DataType>(numClusters, n_features, clusters, n_samples, 50, 1e-5);
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
    // printVecInVec<float>(model->m_clusters, numClusters, n_features, "clusters");
    std::cout << "*********    accuracy  **********" << std::endl;
    std::cout << "model accuracy : " << model->accuracy(label) << std::endl;
    printVecInVec<int>(model->m_sampleClasses, 1, 10, "sampleClasses_10");

    delete model;
}

void launch_kmeans(
    float *data,
    const int n_clusters,
    const int n_samples,
    const int n_features
    )
{
    

    int* cidx = new int[n_clusters];
    float* centroids = new float[n_clusters * n_features];
    srand(time(NULL));
    for (int i = 0; i < n_clusters; i++)
        cidx[i] = rand() % n_samples;

    for (int i = 0; i < n_clusters; i++)
    {
        for (int j = 0; j < n_features; j++)
        {
            centroids[i * n_features + j] = data[cidx[i] * n_features + j];
        }
    }

    Kmeans<float> *model = new KmeansGPUV3<float>(n_clusters, n_features, centroids, n_samples, 50, 1e-5);

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

    delete model;
}

int main()
{
    string file = "/home/gg/Desktop/kmeans/data/test_samples.csv";
    int N = 0;                   // 实际读取的样本数量
    const int n_nums = 100; // 数据中，有100个样本
    const int n_features = 256;    // 每个样本有4个特征
    const int n_classes = 256;
    float *data = new float[n_features * n_nums];
    int *label = new int[n_nums];
    readCoordinate(data, label, n_features, N, file);
    std::cout << "num of samples : " << N << std::endl;

    // 数据初始化

    int cidx[n_classes] = {0};
    srand(time(NULL));
    for (int i = 0; i < n_classes; i++)
        cidx[i] = rand() % n_nums;
    float clusters[n_classes * n_features] = {0};
    for (int i = 0; i < n_classes; i++)
    {
        for (int j = 0; j < n_features; j++)
        {
            clusters[i * n_features + j] = data[cidx[i] * n_features + j];
        }
    }

    // printVecInVec<float>(clusters, n_classes, n_features, "clusters");

    // std::cout << "Using CPU:" << std::endl;
    // timing<float>(data, label, clusters, n_classes, n_features, N, 0);

    std::cout << "Using CUDA:" << std::endl;
    timing<float>(data, label, clusters, n_classes, n_features, N, 1);
    delete[] data;
    delete[] label;
    return 0;
}
