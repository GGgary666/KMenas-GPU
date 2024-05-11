#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "kmeans_base.h"
#include "kmeans_base_gpu.cuh"
#include "error.cuh"

using namespace std;


void readCoordinate(float *data, int *label, const int n_features, int &n) {
    std::ifstream ifs;
    ifs.open("/home/gg/Desktop/kmeans/data/sample_1e6_fea_100_class_4_lable_1_ninfo_8.csv", std::ios::in);
    if (ifs.fail()) {
        std::cout << "No such file or directory: sample_1e6_fea_100_class_4_lable_1_ninfo_8.csv" << std::endl;
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

    std::cout << "********* final clusters**********" << std::endl;
    printVecInVec<float>(model->m_clusters, 4, 100, "clusters");
    std::cout << "*********    accuracy  **********" << std::endl;
    std::cout << "model accuracy : " << model->accuracy(label) << std::endl;
    printVecInVec<int>(model->m_sampleClasses, 1, 10, "sampleClasses_10");

    delete model;
}


int main() {
    int N = 0;
    int n_features = 100;
    const int bufferSize = 1000000 * n_features;
    float *data = new float[bufferSize];
    int *label = new int[bufferSize];
    readCoordinate(data, label, n_features, N);
    std::cout << "num of samples : " << N << std::endl;

    int cidx[] = {1, 3, 6, 8};
    int numClusters = 4;
    float clusters[400] = {0};
    for (int i=0 ; i<numClusters; ++i) {
        for (int j=0; j<n_features; ++j) {
            clusters[i * n_features + j] = data[cidx[i] * n_features + j];
        }
    }
    std::cout << "********* init clusters **********" << std::endl;
    printVecInVec<float>(clusters, 4, 100, "clusters");

    std::cout << "Using CPU:" << std::endl;
    timing(data, label, clusters, numClusters, n_features, N, 0);
    std::cout << "Using CUDA:" << std::endl;
    timing(data, label, clusters, numClusters, n_features, N, 1);

    delete[] data;
    delete[] label;

    return 0;
}