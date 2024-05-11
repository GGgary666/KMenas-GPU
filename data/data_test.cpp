#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_map>

// 函数声明
std::vector<int> mapClusterToLabel(const std::vector<int>& clusterLabels, const std::vector<int>& trueLabels);

int main() {
    // 示例数据：聚类结果和真实标签
    std::vector<int> clusterLabels = {1, 0, 0, 2, 1, 0, 2, 2, 1, 2};
    std::vector<int> trueLabels =    {0, 0, 1, 1, 0, 1, 2, 2, 0, 2};

    // 调用函数进行映射
    std::vector<int> mappedLabels = mapClusterToLabel(clusterLabels, trueLabels);

    // 打印映射后的标签
    std::cout << "映射后的标签：" << std::endl;
    for (int label : mappedLabels) {
        std::cout << label << " ";
    }
    std::cout << std::endl;

    return 0;
}

std::vector<int> mapClusterToLabel(const std::vector<int>& clusterLabels, const std::vector<int>& trueLabels) {
    std::vector<int> mappedLabels = clusterLabels;

    // 遍历每个簇
    for (int clusterNum = 0; clusterNum < 3; ++clusterNum) {
        std::vector<int> clusterIndices;
        // 找到属于当前簇的数据点的索引
        for (size_t i = 0; i < clusterLabels.size(); ++i) {
            if (clusterLabels[i] == clusterNum) {
                clusterIndices.push_back(i);
            }
        }

        // 统计当前簇中真实标签出现的频率
        std::unordered_map<int, int> labelFreq;
        for (int index : clusterIndices) {
            ++labelFreq[trueLabels[index]];
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

    return mappedLabels;
}
