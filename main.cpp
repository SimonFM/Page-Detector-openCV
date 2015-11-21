/*
 * Simon Markham
 * 
 */
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "Utilities.h"
#include "Histograms.cpp"

#define NUM_OF_BINS	4 
#define HALF_SIZE 2
#define THIRD_SIZE 3

using namespace std;
using namespace cv;

// Location of the images in the project
char * bookLoc = "Media/Books/";
char * pageLoc = "Media/Pages/";
char * sampleBlue = "Media/BlueBookPixels.png";

Mat sampleBluePixel;

char * pages[] = {"Page01.jpg", "Page02.jpg",
				  "Page03.jpg", "Page04.jpg",
				  "Page05.jpg", "Page06.jpg",
				  "Page07.jpg", "Page08.jpg",
				  "Page09.jpg", "Page10.jpg",
				  "Page11.jpg", "Page12.jpg",
				  "Page13.jpg"};

char * books[] = {"BookView01.jpg", "BookView02.jpg",
				  "BookView03.jpg", "BookView04.jpg",
				  "BookView05.jpg", "BookView06.jpg",
				  "BookView07.jpg", "BookView08.jpg",
				  "BookView09.jpg", "BookView10.jpg",
				  "BookView11.jpg", "BookView12.jpg",
				  "BookView13.jpg", "BookView14.jpg",
				  "BookView15.jpg", "BookView16.jpg",
				  "BookView17.jpg", "BookView18.jpg",
				  "BookView19.jpg", "BookView20.jpg",
				  "BookView21.jpg", "BookView22.jpg",
				  "BookView23.jpg", "BookView24.jpg",
				  "BookView25.jpg", "BookView26.jpg",
				  "BookView27.jpg", "BookView28.jpg",
				  "BookView29.jpg", "BookView30.jpg",
				  "BookView31.jpg", "BookView32.jpg",
				  "BookView33.jpg", "BookView34.jpg",
				  "BookView35.jpg", "BookView36.jpg",
				  "BookView37.jpg", "BookView38.jpg",
				  "BookView39.jpg", "BookView40.jpg",
			      "BookView41.jpg", "BookView42.jpg",
				  "BookView43.jpg", "BookView44.jpg",
				  "BookView45.jpg", "BookView46.jpg",
				  "BookView47.jpg", "BookView48.jpg",
				  "BookView49.jpg", "BookView50.jpg"};

// gets the Mask of an page so I can only get the page
void getMask(Mat * images, int size, Mat * &result){
	Mat rgb[3], thresh, erodedImage, dilatedImage;
	Mat mask;
	vector<Mat> channels;
	for(int i = 0 ; i < size ; i++){
		split(images[i],rgb);
		threshold(rgb[0],thresh, 0, 225,  CV_THRESH_BINARY | CV_THRESH_OTSU);
		// perform an opening
		erode(thresh,erodedImage,Mat());
		dilate(erodedImage,mask,Mat());
		// mask all the channels
		for(int j = 0 ; j < 3 ; j++){
			Mat temp_1;
			rgb[j].copyTo(temp_1,mask);
			//bitwise_and(mask,rgb[j],temp_1 );
			channels.push_back(temp_1);
		}
		// merge all the channels in to the result.
		merge(channels,result[i]);
		channels.clear();
	}
}

// A function that loads in a file path, the file names, the number of files, and
// a pointer to where those files should be stored and then stores them at that location
void loadImages(char * fileLocation, char ** imageFiles, int size, Mat * &images){
	int number_of_images = sizeof(imageFiles)/sizeof(imageFiles[0]);

	images = new Mat[size];

	// This code snippet was taken from your OpenCVExample and it loads the images
	for(int i = 0 ; i < size ; i++){
		string filename(fileLocation);
		filename.append(imageFiles[i]);
		images[i] = cv::imread(filename, -1);

		if (images[i].empty()) cout << "Could not open " << filename << endl;
		//else cout << "Opened: " << filename << endl;
	}
}

// a function that resizes an image with a given factor
void resize(Mat * image,int size, int factor, Mat * result){
	for(int i = 0; i < size; i++)
		cv::resize(image[i],result[i],cv::Size(image[i].cols/factor,image[i].rows/factor));
}


// A function that displays an array of given images using the imshow function of opencv
void displayImages(string windowName,int size,Mat * original,Mat * images, Mat * backProjected){
	Mat * display = new Mat[size];
	Mat temp;
	for(int i = 0 ; i < size; i++){
		cvtColor(backProjected[i],temp, CV_GRAY2BGR);
		display[i] = JoinImagesHorizontally(original[i],"Original Image",images[i],"Result Image",4);
		display[i] = JoinImagesHorizontally(display[i],"Original Image",temp,"Back Project Image",4);
		imshow(windowName,display[i]);
		waitKey(0);
	}
}

// Greyscales a set of images and places it in the location.s
void greyScaleArray(Mat * toBeGreyScaled, int size, Mat * &result){
	for(int i = 0; i < size; i++){
		cv::cvtColor(toBeGreyScaled[i],result[i],cv::COLOR_BGR2GRAY);
	}
}

// returns an array of all the red channels
void getRedChannels(Mat * input, int size, Mat * &redChannel){
	Mat  channels[3];
	for(int i = 0; i < size; i++){
		split(input[i],channels);
		redChannel[i] = channels[0];
	}
}

// draws lines on an image for a given list of points.
void drawLines(Mat input,std::vector<Point> points, int size){
	int i = 0, j =0;
	for(i = 0,j = i +1; i < size-1; j++,i++)
		line( input,points[i],points[j],Scalar( 0, 0, 255 ),1,8 );
	line( input,points[0],points[size-1],Scalar( 0, 0, 255),1,8 );
	
}

// draws circles on an image for a given list of points.
void drawCircles(Mat input,std::vector<Point> points, int size){
	for(int i = 0; i < size; i++){
		circle( input, points[i], 5, Scalar( 0, 0, 255 ),-1); 
		//cout <<"Hello" << endl;
	}
}

// gets the locations of the white dots in an image
std::vector<Point> getWhiteDotsLocations(Mat image ){
	Mat nonZero ;
	findNonZero(image,nonZero);
	std::vector<Point> result(nonZero.total());
	for (int i = 0; i < nonZero.total(); i++) result[i] = nonZero.at<Point>(i);
	
	return result;
}

// This back projection code was taken from the openCvExample given to us.
void backProjectionAndThreshold(Mat * imagesToProcesse, int size, Mat * &result){
	Mat hls_image,hls_imageBlue;
	Mat backProjProb,binary;

	cvtColor(sampleBluePixel, hls_imageBlue, CV_BGR2HLS);
	ColourHistogram histogram3D(hls_imageBlue,NUM_OF_BINS);

	histogram3D.NormaliseHistogram();
	for(int i = 0; i < size; i++){
		cvtColor(imagesToProcesse[i], hls_image, CV_BGR2HLS);
		backProjProb = StretchImage( histogram3D.BackProject(hls_image) );
		threshold(backProjProb, result[i], 15, 225,  CV_THRESH_BINARY | CV_THRESH_OTSU);
	}
}

// Applies Ostu thresholding to an entire array of images
void applyOtsuThresholding(Mat * images, int size, Mat * result){
	Mat * grey =  new cv::Mat[size];
	greyScaleArray(images, size,grey);
	for(int i = 0 ; i < size; i++)
		threshold(images[i], result[i], 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
}

// Applies Gaussian Filter to the image
void gaussianBlurImage(Mat * images, int size, Mat * result){
	for(int i = 0 ; i < size; i++)
		GaussianBlur( images[i], result[i], Size(3,3), 0, 0, BORDER_DEFAULT );
}

// K means an image (Code from Example)
Mat kmeans_clustering( Mat& image, int k, int iterations ){
	CV_Assert( image.type() == CV_8UC3 );
	// Populate an n*3 array of float for each of the n pixels in the image
	Mat samples(image.rows*image.cols, image.channels(), CV_32F);
	float* sample = samples.ptr<float>(0);
	for(int row=0; row<image.rows; row++)
		for(int col=0; col<image.cols; col++)
			for (int channel=0; channel < image.channels(); channel++)
				samples.at<float>(row*image.cols+col,channel) =
								(uchar) image.at<Vec3b>(row,col)[channel];
	// Apply k-means clustering to cluster all the samples so that each sample
	// is given a label and each label corresponds to a cluster with a particular
	// centre.
	Mat labels;
	Mat centres;
	kmeans(samples, k, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 1, 0.0001),
		iterations, KMEANS_PP_CENTERS, centres );
	// Put the relevant cluster centre values into a result image
	Mat& result_image = Mat( image.size(), image.type() );
	for(int row=0; row<image.rows; row++)
		for(int col=0; col<image.cols; col++)
			for (int channel=0; channel < image.channels(); channel++)
				result_image.at<Vec3b>(row,col)[channel] = (uchar) centres.at<float>(*(labels.ptr<int>(row*image.cols+col)), channel);
	return result_image;
}

// Applies K Means to an array of images and stores them in result
void kMeansImage(Mat * images, int size, Mat * result){
	for(int i = 0; i < size; i++)
		result[i] = kmeans_clustering(images[i],4,1);
}

//gets the bottom left corner in an image
Point getBottomLeftPoint(std::vector<Point> points){
	Point bottomLeft;
	int size = points.size();
	bottomLeft = points[0];
	for(int i = 0; i < size; i++){
		if( bottomLeft.x > points[i].x)	bottomLeft = points[i];
	}
	return bottomLeft;
}

//gets the bottom right circle in an image
Point getTopRightPoint(std::vector<Point> points){
	Point bottomRight;
	int size = points.size();
	bottomRight = points[0];
	for(int i = 0; i < size; i++){
		if( bottomRight.x < points[i].x) bottomRight = points[i];
	}
	return bottomRight;
}

//gets the bottom right circle in an image
Point getBottomRightPoint(std::vector<Point> points){
	Point bottomRight;
	int size = points.size();
	bottomRight = points[0];
	for(int i = 0; i < size; i++){
		if( bottomRight.y < points[i].y) bottomRight = points[i];
	}
	return bottomRight;
}

//gets the top left circle in an image
Point getTopLeftPoint(std::vector<Point> points){
	Point topRight;
	int size = points.size();
	topRight = points[0];
	for(int i = 0; i < size; i++){
		if( topRight.y > points[i].y) 	topRight = points[i];
	}
	return topRight;
}

// Gets the area of a page, given 4 co ordinates
int areaOfPage(int x1, int y1, int x2,  int y2, int x3,  int y3, int x4,  int y4 ){
	int area = (x1*y2 - x2*y1) + (x2*y3 - x3*y2) + (x3*y4 - x4*y3) + (x4*y1 - x1*y4);
	std::cout <<"Area: "<<area <<std::endl;
	return area / 2;
}

// returns the angle between point a and b
float angleBetweenTwoPoints(Point a, Point b){
	float angle = atan2(a.y - b.y, a.x - b.x);
	std::cout <<"Angle: "<<angle <<std::endl;
	return angle;
}
//draws the circles and lines on an image 
void drawLocationOfPage(Mat * backProjectionImages, Mat * images, int size,	std::vector<Point> whiteDots, std::vector<Point> corners){
	for(int i = 0; i < size; i++){
		whiteDots = getWhiteDotsLocations(backProjectionImages[i]);
		corners.push_back( getBottomLeftPoint(whiteDots));
		corners.push_back( getBottomRightPoint(whiteDots));
    	corners.push_back( getTopRightPoint(whiteDots));
		corners.push_back( getTopLeftPoint(whiteDots));
		/*areaOfPage(corners[0].x,corners[0].y,corners[1].x,corners[1].y,
					corners[2].x,corners[2].y,corners[3].x,corners[3].y);*/
		//angleBetweenTwoPoints(corners[0],corners[3]);
	//	angleBetweenTwoPoints(corners[1],corners[2]);

		drawCircles(images[i],corners,corners.size());
		drawLines(images[i],corners,corners.size());
		corners.clear();
	}
}

// Function to run program
int main(int argc, const char** argv){	
	int bookSize = sizeof(books) / sizeof(books[0]);
	int pageSize = sizeof(pages) / sizeof(pages[0]);
	Mat * booksMat = new Mat[bookSize];
	Mat * pagesMat = new Mat[pageSize];

	Mat * resizedBooks = new Mat[bookSize];
	Mat * originalResizedBooks = new Mat[bookSize];
	Mat * resizedPages = new Mat[pageSize];

	Mat * backProjectionImages = new Mat[bookSize];
	Mat * mask = new Mat[bookSize];

	std::vector<Point> whiteDotsLocation;
	std::vector<Point> corners;

	sampleBluePixel = cv::imread(sampleBlue, -1);
	loadImages(bookLoc, books, bookSize, booksMat);
	loadImages(pageLoc, pages, pageSize, pagesMat);
	// resize to improve speed
	resize(booksMat,bookSize, THIRD_SIZE, resizedBooks); 
	resize(booksMat,bookSize, THIRD_SIZE, originalResizedBooks); 
	getMask(resizedBooks,bookSize,mask);

	// backproject and threshold, then draw circles where the corners are
	backProjectionAndThreshold(mask,bookSize,backProjectionImages);
	drawLocationOfPage(backProjectionImages, mask ,bookSize, whiteDotsLocation, corners);
	
	displayImages("Found Corners",bookSize,originalResizedBooks,mask,backProjectionImages);
	//displayImages("Found Corners",bookSize,originalResizedBooks,mask,backProjectionImages);
    return 0;
}