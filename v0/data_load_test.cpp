#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <stdexcept>

using namespace std;

void readCoordinate(float *data, int *label, const int n_features, int &n) {
    std::ifstream ifs;
    ifs.open("/home/gg/Desktop/kmeans/data/test.csv", std::ios::in);
    if (ifs.fail()) {
        std::cout << "No such file or directory: test.csv" << std::endl;
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

int main() {
    const int n_features = 4; // 假设有100个特征
    const int n_nums = 100;
    float data[n_features * n_nums]; // 假设最多读取1000个样本
    int label[n_nums]; // 对应的标签数组
    int n = 0; // 实际读取的样本数量

    // 读取CSV文件
    readCoordinate(data, label, n_features, n);

    // 打印输出读取的数据和标签
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n_features; ++j) {
            std::cout << data[i * n_features + j] << ",";
        }
        std::cout << "Label: " << label[i] << std::endl;
    }

    return 0;
}