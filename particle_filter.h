#ifndef __particle_filter_h
#define __particle_filter_h

#include <string.h>
#include <stdlib.h>
#include <iostream> // to show the result
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <limits.h>
#include <unistd.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core.hpp>
#include <algorithm>
#include <iomanip>
#include <time.h>
#include <math.h>
#include <fstream>
#include <random>

enum MorpologyType { DILATION, EROTION};

class Gaussian
{
private:
    float m_mean;
    float m_stddev;

public:
    inline float getMean() { return m_mean; }
    inline void setMean(float value) { m_mean = value; }

    inline float getStandardDeviation() { return m_stddev; }
    inline void setStandardDeviation(float value) { m_stddev = value; }

    Gaussian(float mean, float standard_deviation)
    {
        m_mean = mean;
        m_stddev = standard_deviation;
    }
};

//
class Particle
{
private:
    int m_x;
    int m_y;
    int m_w;
    int m_h;
    float m_weight;

public:
    inline int getX() const { return m_x; }
    inline void setX(int value) { m_x = value; }

    inline int getY() const { return m_y; }
    inline void setY(int value) { m_y = value; }

    inline int getW() const { return m_w; }
    inline void setW(int value) { m_w = value; }

    inline int  getH() const { return m_h; }
    inline void setH(int value) { m_h = value; }

    inline float getWeight() const { return m_weight; }
    inline void setWeight(float value) { m_weight = value; }

    Particle(int x, int y, int w, int h, float weight)
    {
        m_x = x;
        m_y = y;
        m_w = w;
        m_h = h;
        m_weight = weight;
    }

    /*
        //for descending order sorting
        bool operator < (const Particle& prt) const
        {
            return (weight >= prt.weight);
        } */
};



class ParticleFilter
{
private:
    //#define NUMBER_OF_PARTICLES 7
    //#define NUM_OF_ITERATONS 100
    Gaussian *gaussian;
    float RED;
    std::vector<Particle *> particles;
    std::vector<Particle *> particlesNew;
    int imageHeight, imageWidth;
    int particleWidth, particleHeight;
    cv::Mat image, dst;
    cv::Mat histref1, histref2;
    char winName[20];
    int NUMBER_OF_PARTICLES;
    int NUM_OF_ITERATONS;

public:
    ParticleFilter(const char *filename, int nParticles, int nIters)
    {
        RED = 136;
        NUMBER_OF_PARTICLES = nParticles;
        NUM_OF_ITERATONS = nIters;
        image = imread(filename, cv::IMREAD_COLOR);
        cv::Mat imgHSV;
        image.copyTo(imgHSV);
        cvtColor(image, imgHSV, CV_BGR2HSV);
        dst = redFilter(imgHSV);
    }

    void run();

    Gaussian *gaussParameters(cv::Mat data);
    cv::Mat getRedMeans();

    float calcGaussian(float x, float m, float s);

    // creates random particles with equal weights
    void initializeParticles();

    // traverse all particles and checks the particle's rgb value against our model's rgb values
    // and update the weights
    void updateWeights();

    // returns mean of red channels as an vector of [red]
    float getRIntensityMean(cv::Mat image);

    // Euclidian Distance between our model and the mean values of each particle
    float calcDistance(cv::Mat GRvector);

    // creates new particles around the highly weighted particles
    void resample();

    // displays particles on image
    void showParticles(int window_index);

    // sort according to weights
    void sortParticlesDescending();

    // normalize particle weights
    void normalize();

    // filter out only red channels using an HSV image
    cv::Mat redFilter(const cv::Mat &src);

    // distort x or y by a standart deviation
    void distort(Particle *p, int &x, int &y);
};

#endif // __particle_filter_h
