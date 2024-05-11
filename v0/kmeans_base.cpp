#include <cmath>
#include <iostream>
#include <vector>
#include <unordered_map>
#include "kmeans_base.h"

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





