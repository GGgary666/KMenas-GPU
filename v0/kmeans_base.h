#pragma once
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