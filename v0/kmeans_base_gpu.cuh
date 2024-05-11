#ifndef CUDA_HEADER_FILE_H
#define CUDA_HEADER_FILE_H
#include "error.cuh"
#include "kmeans_base.h"

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

#endif