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

class Gaussian{
	private:
	float mean;
	float standard_deviation;

	public:
	void setMean(float mean){this->mean=mean;}
	void setStandardDeviation(float standard_deviation){this->standard_deviation=standard_deviation;}
	float getMean(){return mean;}
	float getStandardDeviation(){return standard_deviation;}

	Gaussian(float mean,float standard_deviation){
		this->mean=mean;
		this->standard_deviation=standard_deviation;
	}
};
//
class Particle{
	private:
	int x;
	int y;
	int w;
	int h;
	float weight;
	public:
		int getX() {return x;}
		void setX(int x){this->x=x;}
		int getY() {return y;}
		void setY(int y){this->y=y;}
		int getW() {return w;}
		void setW(int w){this->w=w;}
		int  getH(){return h;}
		void setH(int h){this->h=h;}
		float getWeight(){return weight;}
		void setWeight(float weight){this->weight=weight;}

		Particle(int x,int y,int w,int h,float weight){
			this->x=x;
			this->y=y;
			this->w=w;
			this->h=h;
			this->weight=weight;
		}

		/*
		//for descending order sorting
	    bool operator < (const Particle& prt) const
	    {
	        return (weight >= prt.weight);
	    } */

};

using namespace std;
using namespace cv;

//#define NUMBER_OF_PARTICLES 7
//#define NUM_OF_ITERATONS 100
Gaussian *gaussian;
float RED=136;
vector<Particle*> particles;
vector<Particle*> particlesNew;
int imageHeight,imageWidth;
int particleWidth,particleHeight;
Mat image,dst;
Mat histref1, histref2;
char winName[20];
int NUMBER_OF_PARTICLES;
int NUM_OF_ITERATONS;

Gaussian* gaussParameters(Mat data);
Mat getRedMeans();

float calcGaussian(float x, float m, float s);
//creates random particles with equal weights
void InitilizeParticles();
//traverse all particles and checks the particle's rgb value against our model's rgb values
//and update the weights
void updateWeights();
//returns mean of red channels as an vector of [red]
float getRIntensityMean(cv::Mat image);
//Euclidian Distance between our model and the mean values of each particle
float calcDistance(Mat GRvector);
//creates new particles around the highly weighted particles
void resample();
//displays particles on image
void showParticles(int window_index);
//sort according to weights
void sortParticlesDescending();
//normalize particle weights
void normalize();
//filter out only red channels using an HSV image
Mat redFilter(const Mat& src);
//distort x or y by a standart deviation
void distort(Particle* p, int&x , int& y);

vector<int> regionQuery(vector<Point> *points, Point *point, float eps);
vector<vector<Point> > DBSCAN_points(vector<Point> *points, float eps, unsigned int minPts);

std::ofstream out;

//Size frameSize(vcap.get(CV_CAP_PROP_FRAME_WIDTH)-filter.cols -1, vcap.get(CV_CAP_PROP_FRAME_HEIGHT)- filter.rows -1);
    //VideoWriter videoWriter("evolution.avi", -1, 10, Size(640,351), true); //ask user to choose codec

std::default_random_engine generator;
std::normal_distribution<double> distribution(0.0,1.0);



int main( int numArgs, char *argv[])
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
	out.open ("out.txt", std::ofstream::out | std::ofstream::app);
	//initialize the seed for random generator
	srand(time(NULL));

	/*if(!videoWriter.isOpened())
	{
		cout << "ERROR: Failed to open video for writing\n";
		return -1;
	}*/
	NUMBER_OF_PARTICLES=atoi(argv[2]);
	NUM_OF_ITERATONS=atoi(argv[3]);
    image = imread(argv[1], IMREAD_COLOR);
    Mat imgHSV;
    image.copyTo(imgHSV);
    cvtColor(image, imgHSV, CV_BGR2HSV);
    dst = redFilter(imgHSV);

   // std::default_random_engine generator;
   // std::normal_distribution<double> distribution(5.0,2.0);


    namedWindow("originalImage",CV_WINDOW_NORMAL);
    imshow("originalImage",image);

    namedWindow("After HSV",CV_WINDOW_NORMAL);
    imshow("After HSV",dst);

    waitKey(0);

    imageHeight=image.rows;
    imageWidth=image.cols;
	//scale the particle width and height
    particleWidth=imageWidth/30;
    particleHeight=particleWidth*3;
    int windowId=1;
    InitilizeParticles();
    showParticles(windowId++);

    for(int itrIndex =0; itrIndex<NUM_OF_ITERATONS;itrIndex++)
    {
		updateWeights();
	    normalize();
	    sortParticlesDescending();
		resample();
		//showParticles(windowId++);
		cout << itrIndex+1 << endl;
    }
    showParticles(windowId++);

    vector<vector<Point> > points;
    vector<Point> subpoints;
    for(int i=0;i<NUMBER_OF_PARTICLES;i++){
		 Particle* p = particles[i];
		 Point kp(p->getX(),p->getY());
		 subpoints.push_back(kp);
    }

    // (epsilon = diameter, minpoints)
    points =  DBSCAN_points(&subpoints,40.0,1);
    cout << points.size()-1 << endl;

    particles.clear();

    //QT q graphics scene,git cmake , temporal domain, NN tracking, 2 ellipse model, center distortion and particle, training set label,

    for(unsigned int i=1;i<points.size();i++){
    	vector<Point> kpv = points[i];
    	//Prepare lines found for k-means clustering
        /*Mat particle_points(kpv.size(),2,CV_32F);*/
        float sumx=0, sumy=0;
    	for (unsigned int  j= 0;j < kpv.size(); j++){
    		/*particle_points.at<float>(j,0)=kpv[j].x;
    		particle_points.at<float>(j,1)=kpv[j].y; */
    		sumx += kpv[j].x;
    		sumy += kpv[j].y;
    	}

    	cout << "x:" << sumx/kpv.size() << " y: " << sumy/kpv.size() << endl;
    	Particle *particle=new Particle(sumx/kpv.size(),sumy/kpv.size(),particleWidth,particleHeight,0.1);
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

void showParticles(int windowId){
    int size = particles.size();
    Mat img;
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
	sprintf(winName,"Window%d",windowId);
    imshow(winName, img);
    waitKey(0);
}

void InitilizeParticles(){
	for (int i = 0; i < NUMBER_OF_PARTICLES; ++i)
	{
		int  x=(rand() % (imageWidth- particleWidth));
	    int  y=(rand() % (imageHeight- particleHeight));
        float w=1.0/NUMBER_OF_PARTICLES;
        Particle *particle=new Particle(x,y,particleWidth,particleHeight,w);
        particles.push_back(particle);
	}
}

//updates the weights
void updateWeights(){
    for (size_t i = 0; i < particles.size(); ++i)
    {
   	 int x= particles[i]->getX();
   	 int y= particles[i]->getY();
     // dst is a binary image containing only the red parts
   	 Mat roi = dst(Rect(x,y,particleWidth,particleHeight));
   	 float r_intensityMean = getRIntensityMean(roi);

   	 //if(r_intensityMean>10) r_intensityMean=0.98;
   	 particles[i]->setWeight(r_intensityMean + 0.01);
    }
}

void sortParticlesDescending(){
    //cout << "sorting..." << endl ;
    for(unsigned int i=0;i<particles.size()-1;i++){
         for(unsigned int j=0;j<particles.size()-1;j++){
        	 if(particles[i]->getWeight() < particles[j]->getWeight())
        	 {
        	    Particle * temp = particles[i];
        	    particles[i] = particles[j];
        	    particles[j] = temp;
        	 }
         }
    }
}

void normalize(){

	float total_weight=0;
    for (unsigned int i = 0; i < particles.size(); ++i){
    	total_weight += particles[i]->getWeight();
    }

    //normalize the probabilities so that they sum to 1
    float sum=0;
    for (unsigned int i = 0; i < particles.size(); ++i){
    	particles[i]->setWeight(particles[i]->getWeight()/total_weight);
    	sum += particles[i]->getWeight();
    	//out << particles[i]->getWeight() << ";" << flush;
    }
    //cout << "sum after normalization :  " << sum << endl;
}

void resample(){
	//cout << "samples chosen : " ;
	for(int j=0;j<NUMBER_OF_PARTICLES;j++){
      float rand_number =  (float)(rand() % 100)/100 ;
      float sum=0;
      int i=NUMBER_OF_PARTICLES-1;
      for(; i>=0; i--){
    	  sum += particles[i]->getWeight();
          if(sum >= rand_number) break;
	  }

      if(i<0){
    	  cout << "error in random number";
    	  return;
      }

      //cout <<  i << " ";
      Particle * p_to_distort = particles[i];
      int newX, newY;
      distort(p_to_distort,newX,newY);
      Particle * p_new = new Particle(newX,newY,p_to_distort->getW(),p_to_distort->getH(),1.0/NUMBER_OF_PARTICLES);
      particlesNew.push_back(p_new);
	}

	for(int j=0;j<NUMBER_OF_PARTICLES;j++){
		particles[j] = particlesNew.back();
		particlesNew.pop_back();
	}
}
//Euclidian Distance between our model and the mean values of each particle
// closer the points lower the distance
float calcDistance(Mat GRvector){
	float distance = pow(GRvector.at<float>(0,0)-RED,2);
	return distance;
}

float getRIntensityMean(cv::Mat image){
  //cv::Mat r_val = Mat::zeros(cvSize(1,1), CV_32FC1);
  float mean=0.0;
  for(int rowImg=1;rowImg<image.rows-1;rowImg++){
    for(int colImg=1;colImg<image.cols-1;colImg++){
        int hue=image.at<uchar>(rowImg,colImg);
        mean+= (hue ==255) ? 1.0 : 0.0;
    }
  }
  return mean;
}

Gaussian* gaussParameters(Mat data){
	float sum=0;
	float mean;
	float variance;
	for (int i = 0; i < data.cols; ++i) {
		sum+=data.at<float>(0,i);
	}
	mean=sum/data.cols;
	for (int j = 0;  j < data.cols; ++j) {
		variance+=pow(data.at<float>(0,j)-mean,2);
	}
	return new Gaussian(mean,sqrt(variance/data.cols));
}

// gaussian means for NT pictures
Mat getRedMeans(){
	 int NT=7;
	 Mat means=Mat::zeros(1,NT,CV_32FC1);
     char file_name [20];

	 for(int i=0;i<NT;i++){
		sprintf(file_name,"%d_.jpg",i+1);
		float mean=getRIntensityMean(imread(file_name));
		means.at<float>(0,i)=mean;
	 }
	return means;
}

//given mean(m) and standart deviation(s) gets the normal probability of x
float calcGaussian(float x, float m, float s){
    static const float inv_sqrt_2pi = 0.3989422804014327;
    float a = (x - m) / s;
    return inv_sqrt_2pi / s * std::exp(-0.5f * a * a);
}

Mat morphology(const Mat& image, MorpologyType type, int size){
	Mat element = getStructuringElement(MORPH_ELLIPSE,
										Size(2*size + 1, 2*size+1),
										Point(size, size));
	Mat result;
	if(type == DILATION)
		dilate(image, result, element);
	else
		erode(image, result, element);
	return result;
}

//filter only red pixels
Mat redFilter(const Mat& src)
{
    Mat dstA,dstB,dest;
    //Mat result=redFilter(imgHSV);
	inRange(src, Scalar(0, 135, 135), Scalar(20, 255, 255), dstA);
	inRange(src, Scalar(159, 135, 135), Scalar(179, 255, 255), dstB);
	bitwise_or(dstA, dstB, dest);
	dest = morphology(dest, DILATION, 10);
	//dest = morphology(dest, EROTION, 1);
	return dest;
}

void distort(Particle* p, int&x , int& y){
	try {

		int dx=(int)distribution(generator);
		int dy=(int)distribution(generator);

		x=p->getX();
		y=p->getY();
		int newx = x+dx;
		int newy = y+dy;
		if(newx < imageWidth-particleWidth && newx>=0)   x=newx;
		if(newy < imageHeight-particleHeight && newy>=0) y=newy;
	}
    catch(int e){cout << "HATA" << e;}
}

/* DBSCAN - density-based spatial clustering of applications with noise */

vector<vector<Point> > DBSCAN_points(vector<Point> *points, float eps, unsigned int minPts)
{
vector<vector<Point> > clusters;
vector<bool> clustered;
vector<int> noise;
vector<bool> visited;
vector<int> neighborPts;
vector<int> neighborPts_;
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
clusters.push_back(vector<Point>()); //will stay empty?

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
            clusters.push_back(vector<Point>());

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

vector<int> regionQuery(vector<Point> *points, Point *point, float eps)
{
float dist;
vector<int> retKeys;
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




