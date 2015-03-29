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

#include "particle_filter.h"

std::vector<int> regionQuery(std::vector<cv::Point> *points, cv::Point *point, float eps);
std::vector<std::vector<cv::Point> > DBSCAN_points(std::vector<cv::Point> *points, float eps, unsigned int minPts);

std::ofstream out;

//Size frameSize(vcap.get(CV_CAP_PROP_FRAME_WIDTH)-filter.cols -1, vcap.get(CV_CAP_PROP_FRAME_HEIGHT)- filter.rows -1);
//VideoWriter videoWriter("evolution.avi", -1, 10, Size(640,351), true); //ask user to choose codec

std::default_random_engine generator;
std::normal_distribution<double> distribution(0.0, 1.0);

int main(int numArgs, char *argv[])
{
    /*for (int i = 0; i < 100; ++i)
    {
        int number = (int)distribution(generator);
        //int number2=(int)number;
        cout<<number<<endl;
    }*/

    /*cv::Mat mean = cv::Mat::zeros(1,1,CV_64FC1);
    cv::Mat sigma= cv::Mat::ones(1,1,CV_64FC1);
    cv::Mat matrix2xN(2,1,CV_64FC1);
    cv::randn(matrix2xN,  mean, sigma);
    cout<<(int)matrix2xN.at<float>(1,1)<<endl;
    cout<<(int)matrix2xN.at<float>(0,1)<<endl;
    return 0;*/
    /*
    Mat temp=getRedMeans();
    cout<<temp<<endl;
    gaussian=gaussParameters(temp);
    cout<<"Mean: "<<gaussian->getMean()<<endl;
    cout<<"variance: "<<gaussian->getStandardDeviation()<<endl; */

    //return 0;
    out.open("out.txt", std::ofstream::out | std::ofstream::app);
    //initialize the seed for random generator
    srand(time(NULL));

    /*if(!videoWriter.isOpened())
    {
        cout << "ERROR: Failed to open video for writing\n";
        return -1;
    }*/

    ParticleFilter PF((const char *)argv[1], atoi(argv[2]), atoi(argv[3]));
    PF.run();


}

void ParticleFilter::run()
{
    // std::default_random_engine generator;
    // std::normal_distribution<double> distribution(5.0,2.0);

    cv::namedWindow("originalImage", CV_WINDOW_NORMAL);
    imshow("originalImage", image);

    cv::namedWindow("After HSV",CV_WINDOW_NORMAL);
    imshow("After HSV", dst);

    cv::waitKey(0);

    imageHeight = image.rows;
    imageWidth = image.cols;

    // scale the particle width and height
    particleWidth = imageWidth / 30;
    particleHeight = particleWidth*3;
    int windowId = 1;

    initializeParticles();
    showParticles(windowId++);

    for(int itrIndex=0; itrIndex<NUM_OF_ITERATONS; itrIndex++)
    {
        updateWeights();
        normalize();
        sortParticlesDescending();
        resample();
        //showParticles(windowId++);
        std::cout << itrIndex+1 << std::endl;
    }
    showParticles(windowId++);

    std::vector<std::vector<cv::Point> > points;
    std::vector<cv::Point> subpoints;
    for(int i=0; i<NUMBER_OF_PARTICLES; i++)
    {
        Particle *p = particles[i];
        cv::Point kp(p->getX(), p->getY());
        subpoints.push_back(kp);
    }

    // (epsilon = diameter, minpoints)
    points = DBSCAN_points(&subpoints,40.0,1);
    std::cout << points.size()-1 << std::endl;

    particles.clear();

    //QT q graphics scene, git cmake, temporal domain, NN tracking, 2 ellipse model, center distortion and particle, training set label,

    for(unsigned int i=1; i<points.size(); i++)
    {
        std::vector<cv::Point> kpv = points[i];
        //Prepare lines found for k-means clustering
        /*Mat particle_points(kpv.size(),2,CV_32F);*/
        float sumx=0, sumy=0;
        for (unsigned int  j= 0;j < kpv.size(); j++)
        {
            /*particle_points.at<float>(j,0)=kpv[j].x;
            particle_points.at<float>(j,1)=kpv[j].y; */
            sumx += kpv[j].x;
            sumy += kpv[j].y;
        }

        std::cout << "x:" << sumx/kpv.size() << " y: " << sumy/kpv.size() << std::endl;
        Particle *particle = new Particle(sumx/kpv.size(), sumy/kpv.size(), particleWidth, particleHeight, 0.1);
        particles.push_back(particle);

        /*int clusterCount = 1, attempts = 300;
        Mat labels, centers;
        try{
            kmeans(particle_points, clusterCount, labels,
                    TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001),
                    attempts, KMEANS_RANDOM_CENTERS, centers);
            cout << centers.at<float>(0,0) << endl;
        }catch(exception& e){
            cout << "Couldn't determine time" << endl ;
            return -1;
        } */
    }

    showParticles(windowId++);
}

void ParticleFilter::showParticles(int windowId)
{
    int size = particles.size();
    cv::Mat img;
    image.copyTo(img);
    for (int i = 0; i < size; ++i)
    {
        int x= particles[i]->getX();
        int y= particles[i]->getY();
        int x_end = x+particleWidth;
        int y_end = y+particleHeight;
        rectangle(img,
                  cvPoint(x,y),
                  cvPoint(x_end,y_end),
                  cvScalar(255,0,0),
                  1);
    }

    //string winname = "image" + to_string(j);
    //videoWriter << img;
    sprintf(winName, "Window%d", windowId);
    cv::imshow(winName, img);
    cv::waitKey(0);
}

void ParticleFilter::initializeParticles()
{
    for (int i = 0; i < NUMBER_OF_PARTICLES; ++i)
    {
        int x = (rand() % (imageWidth- particleWidth));
        int y = (rand() % (imageHeight- particleHeight));
        float w = 1.0/NUMBER_OF_PARTICLES;
        Particle *particle = new Particle(x,y,particleWidth,particleHeight,w);
        particles.push_back(particle);
    }
}

//updates the weights
void ParticleFilter::updateWeights()
{
    for (size_t i = 0; i < particles.size(); ++i)
    {
        int x = particles[i]->getX();
        int y = particles[i]->getY();
        // dst is a binary image containing only the red parts
        cv::Mat roi = dst(cv::Rect(x, y, particleWidth, particleHeight));
        float r_intensityMean = getRIntensityMean(roi);

        //if(r_intensityMean>10) r_intensityMean=0.98;
        particles[i]->setWeight(r_intensityMean + 0.01);
    }
}

void ParticleFilter::sortParticlesDescending()
{
    //cout << "sorting..." << endl ;
    for(unsigned int i=0;i<particles.size()-1;i++)
    {
        for(unsigned int j=0;j<particles.size()-1;j++)
        {
            if(particles[i]->getWeight() < particles[j]->getWeight())
            {
                Particle *temp = particles[i];
                particles[i] = particles[j];
                particles[j] = temp;
            }
        }
    }
}

void ParticleFilter::normalize()
{
    float total_weight=0;
    for (unsigned int i = 0; i < particles.size(); ++i)
    {
        total_weight += particles[i]->getWeight();
    }

    //normalize the probabilities so that they sum to 1
    float sum=0;
    for (unsigned int i = 0; i < particles.size(); ++i)
    {
        particles[i]->setWeight(particles[i]->getWeight()/total_weight);
        sum += particles[i]->getWeight();
        //out << particles[i]->getWeight() << ";" << flush;
    }
    //cout << "sum after normalization :  " << sum << endl;
}

void ParticleFilter::resample()
{
    //cout << "samples chosen : " ;
    for(int j=0;j<NUMBER_OF_PARTICLES;j++)
    {
        float rand_number = (float)(rand() % 100) / 100;
        float sum = 0;
        int i = NUMBER_OF_PARTICLES-1;
        for(; i>=0; i--)
        {
            sum += particles[i]->getWeight();
            if(sum >= rand_number)
                break;
        }

        if(i<0)
        {
            std::cout << "error in random number";
            return;
        }

        //std::cout <<  i << " ";
        Particle *p_to_distort = particles[i];
        int newX, newY;
        distort(p_to_distort,newX,newY);
        Particle *p_new = new Particle(newX, newY, p_to_distort->getW(), p_to_distort->getH(), 1.0/NUMBER_OF_PARTICLES);
        particlesNew.push_back(p_new);
    }

    for(int j=0; j<NUMBER_OF_PARTICLES; j++)
    {
        particles[j] = particlesNew.back();
        particlesNew.pop_back();
    }
}

// Euclidian Distance between our model and the mean values of each particle
// closer the points lower the distance
float ParticleFilter::calcDistance(cv::Mat GRvector)
{
    float distance = pow(GRvector.at<float>(0,0)-RED,2);
    return distance;
}

float ParticleFilter::getRIntensityMean(cv::Mat image)
{
    //cv::Mat r_val = Mat::zeros(cvSize(1,1), CV_32FC1);
    float mean = 0.0;
    for(int rowImg=1; rowImg<image.rows-1; rowImg++)
    {
        for(int colImg=1; colImg<image.cols-1; colImg++)
        {
            int hue = image.at<uchar>(rowImg, colImg);
            mean += (hue ==255) ? 1.0 : 0.0;
        }
    }
    return mean;
}

Gaussian *ParticleFilter::gaussParameters(cv::Mat data)
{
    float sum = 0;
    float mean;
    float variance;
    for (int i = 0; i < data.cols; ++i)
    {
        sum += data.at<float>(0, i);
    }

    mean = sum/data.cols;
    for (int j = 0;  j < data.cols; ++j)
    {
        variance += pow(data.at<float>(0,j)-mean, 2);
    }
    return new Gaussian(mean, sqrt(variance/data.cols));
}

// gaussian means for NT pictures
cv::Mat ParticleFilter::getRedMeans()
{
    int NT=7;
    cv::Mat means = cv::Mat::zeros(1,NT,CV_32FC1);
    char file_name [20];

    for(int i=0;i<NT;i++)
    {
        sprintf(file_name,"%d_.jpg",i+1);
        float mean = getRIntensityMean(cv::imread(file_name));
        means.at<float>(0, i) = mean;
    }
    return means;
}

// given mean(m) and standart deviation(s) gets the normal probability of x
float calcGaussian(float x, float m, float s)
{
    static const float inv_sqrt_2pi = 0.3989422804014327;
    float a = (x - m) / s;
    return inv_sqrt_2pi / s * std::exp(-0.5f * a * a);
}

cv::Mat morphology(const cv::Mat &image, MorpologyType type, int size)
{
    cv::Mat element = getStructuringElement(
                cv::MORPH_ELLIPSE,
                cv::Size(2*size + 1, 2*size+1),
                cv::Point(size, size)
                );

    cv::Mat result;
    if(type == DILATION)
        dilate(image, result, element);
    else
        erode(image, result, element);

    return result;
}

//filter only red pixels
cv::Mat ParticleFilter::redFilter(const cv::Mat &src)
{
    cv::Mat dstA, dstB, dest;
    //Mat result=redFilter(imgHSV);
    cv::inRange(src, cv::Scalar(0, 135, 135), cv::Scalar(20, 255, 255), dstA);
    cv::inRange(src, cv::Scalar(159, 135, 135), cv::Scalar(179, 255, 255), dstB);
    cv::bitwise_or(dstA, dstB, dest);
    dest = morphology(dest, DILATION, 10);
    //dest = morphology(dest, EROTION, 1);
    return dest;
}

void ParticleFilter::distort(Particle *p, int &x, int &y)
{
    try
    {
        int dx = (int)distribution(generator);
        int dy = (int)distribution(generator);

        x = p->getX();
        y = p->getY();
        int newx = x+dx;
        int newy = y+dy;
        if(newx < imageWidth-particleWidth && newx>=0)
            x = newx;

        if(newy < imageHeight-particleHeight && newy>=0)
            y = newy;
    }
    catch(cv::Exception &e)
    {
        std::cout << "HATA" << e.msg;
    }
}

/* DBSCAN - density-based spatial clustering of applications with noise */

std::vector<std::vector<cv::Point> > DBSCAN_points(std::vector<cv::Point> *points, float eps, unsigned int minPts)
{
    std::vector<std::vector<cv::Point> > clusters;
    std::vector<bool> clustered;
    std::vector<int> noise;
    std::vector<bool> visited;
    std::vector<int> neighborPts;
    std::vector<int> neighborPts_;
    int c;

    int noKeys = points->size();

    //init clustered and visited
    for(int k = 0; k < noKeys; k++)
    {
        clustered.push_back(false);
        visited.push_back(false);
    }

    //C =0;
    c = 0;
    clusters.push_back(std::vector<cv::Point>()); //will stay empty?

    //for each unvisted point P in dataset points
    for(int i = 0; i < noKeys; i++)
    {
        if(!visited[i])
        {
            //Mark P as visited
            visited[i] = true;
            neighborPts = regionQuery(points,&points->at(i),eps);
            if(neighborPts.size() < minPts)
                //Mark P as Noise
                noise.push_back(i);
            else
            {
                clusters.push_back(std::vector<cv::Point>());

                // expand cluster
                // add P to cluster c
                clusters[c].push_back(points->at(i));
                c++;
                //for each point P' in neighborPts
                for(unsigned int j = 0; j < neighborPts.size(); j++)
                {
                    //if P' is not visited
                    if(!visited[neighborPts[j]])
                    {
                        //Mark P' as visited
                        visited[neighborPts[j]] = true;
                        neighborPts_ = regionQuery(points,&points->at(neighborPts[j]),eps);
                        if(neighborPts_.size() >= minPts)
                        {
                            neighborPts.insert(neighborPts.end(),neighborPts_.begin(),neighborPts_.end());
                        }
                    }
                    // if P' is not yet a member of any cluster
                    // add P' to cluster c
                    if(!clustered[neighborPts[j]])
                        clusters[c].push_back(points->at(neighborPts[j]));
                }
            }

        }
    }
    return clusters;
}

std::vector<int> regionQuery(std::vector<cv::Point> *points, cv::Point *point, float eps)
{
    float dist;
    std::vector<int> retKeys;
    for(unsigned int i = 0; i< points->size(); i++)
    {
        dist = sqrt(pow((point->x - points->at(i).x),2)+pow((point->y - points->at(i).y),2));
        if(dist <= eps && dist != 0.0f)
        {
            retKeys.push_back(i);
        }
    }
    return retKeys;
}




