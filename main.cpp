/*
 * Simon Markham
 * 
 */
#pragma region INCLUDES

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
#pragma endregion INCLUDES

using namespace std;
using namespace cv;

#pragma region DEFINES

#define NUM_OF_BINS	4 
#define NUM_OF_CORNERS	4
#define HALF_SIZE 2
#define THIRD_SIZE 3

#pragma endregion DEFINES


#pragma region IMAGE LOCATIONS
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
#pragma endregion IMAGE LOCATIONS

// Resize and Load
#pragma region IMAGE FUNCTIONS
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

#pragma endregion IMAGE FUNCTIONS

// Drawing Lines and Circles
#pragma region DRAWING
// draws lines on an image for a given list of points.
void drawLines(Mat input,std::vector<Point> points){
	line( input,points[0],points[1],Scalar( 0, 0, 255 ),1,8 );
	line( input,points[1],points[3],Scalar( 0, 0, 255),1,8 );
	line( input,points[2],points[3],Scalar( 0, 0, 255),1,8 );
	line( input,points[2],points[0],Scalar( 0, 0, 255),1,8 );
}

// draws circles on an image for a given list of points.
void drawCircles(Mat input,std::vector<Point> points, int size){
	circle( input, points[0], 5, Scalar( 0, 0, 0),-1); 
	circle( input, points[1], 5, Scalar( 0, 255, 0 ),-1); 
	circle( input, points[2], 5, Scalar( 0, 0, 255 ),-1); 
	circle( input, points[3], 5, Scalar( 255, 0, 0 ),-1); 
}

// A function that displays an array of given images using the imshow function of opencv
void displayImages(string windowName,int size,Mat * original,Mat * images, Mat * backProjected){
	Mat * display = new Mat[size];
	Mat * display1 = new Mat[size];
	Mat * display2 = new Mat[size];
	Mat * display3 = new Mat[size];


	Mat temp;
	resize(original,size, THIRD_SIZE, display1); 
	resize(images,size, THIRD_SIZE, display2); 
	resize(backProjected,size, THIRD_SIZE, display3); 

	for(int i = 0 ; i < size; i++){
		cvtColor(display3[i],temp, CV_GRAY2BGR);
		display[i] = JoinImagesHorizontally(display1[i],"Original Image",display2[i],"Result Image",4);
		display[i] = JoinImagesHorizontally(display[i],"Original Image",temp,"Back Project Image",4);
		imshow(windowName,display[i]);
		waitKey(0);
	}
}
#pragma endregion DRAWING

// greyScaleArray
// backProjectionAndThreshold
// applyOtsuThresholding
// gaussianBlurImage
// kmeans_clustering 
// getMask
// getRedChannels
// Alot of the code in here was based off the code given to us in the 
// OpenCVSample on Blackboard
#pragma region OPERATIONS

// Greyscales a set of images and places it in the location.s
void greyScaleArray(Mat * toBeGreyScaled, int size, Mat * &result){
	for(int i = 0; i < size; i++)
		cv::cvtColor(toBeGreyScaled[i],result[i],cv::COLOR_BGR2GRAY);
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
			channels.push_back(temp_1);
		}
		// merge all the channels in to the result.
		merge(channels,result[i]);
		channels.clear();
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
#pragma endregion OPERATIONS

// getWhiteDotsLocations
// getBottomLeftPoint
// getTopRightPoint
// getBottomRightPoint
// getTopLeftPoint
// areaOfPage
// angleBetweenTwoPoints
#pragma region POINTS

// gets the locations of the white dots in an image
std::vector<Point> getWhiteDotsLocations(Mat image ){
	Mat nonZero ;
	findNonZero(image,nonZero);
	std::vector<Point> result(nonZero.total());
	for (int i = 0; i < nonZero.total(); i++) result[i] = nonZero.at<Point>(i);
	
	return result;
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
	Point topRight;
	int size = points.size();
	topRight = points[0];
	for(int i = 0; i < size; i++){
		if( topRight.x < points[i].x) topRight = points[i];
	}
	return topRight;
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
	Point topLeft;
	int size = points.size();
	topLeft = points[0];
	for(int i = 0; i < size; i++){
		if( topLeft.y > points[i].y) 	topLeft = points[i];
	}
	return topLeft;
}

// Gets the area of a page, given 4 co ordinates
int areaOfPage(int x1, int y1, int x2,  int y2, int x3,  int y3, int x4,  int y4 ){
	int area = (x1*y2 - x2*y1) + (x2*y3 - x3*y2) + (x3*y4 - x4*y3) + (x4*y1 - x1*y4);
	std::cout <<"Area: "<<area <<std::endl;
	return area / 2;
}

// returns the angle between point a and b
double angleBetweenTwoPoints(Point a, Point b){
	double angle = atan2(a.y - b.y, a.x - b.x);
	std::cout <<"Angle: "<<angle <<std::endl;
	return angle;
}

#pragma endregion POINTS

// Contains functions for cropping the images and getting the points
#pragma region CROPPING
//draws the circles and lines on an image 
void drawLocationOfPage(Mat * backProjectionImages, Mat * images, int size,	std::vector<Point> *  whiteDots){
	std::vector<Point> temp;	
	Point * p = new Point[NUM_OF_CORNERS];
	for(int i = 0; i < size; i++){
		temp = getWhiteDotsLocations(backProjectionImages[i]);
		p[0] =  getTopLeftPoint(temp);
		p[1] = getTopRightPoint(temp);
		p[2] = getBottomLeftPoint(temp);
		p[3] = getBottomRightPoint(temp);

		for(int j = 0; j < NUM_OF_CORNERS; j++)	whiteDots[i].push_back(p[j]);

		drawCircles(images[i],whiteDots[i],whiteDots[i].size());
		drawLines(images[i],whiteDots[i]);
	}
}

// crops an image given a set of points and stored the cropped image in result.
void cropImage(Mat image, int x ,int y, int xWidth, int yWidth, Mat result){
	result = image(Rect(x ,y ,xWidth ,yWidth));
}

//
double distance(Point p1, Point p2){
	return (int) sqrt( ((p2.x - p1.x) * (p2.x - p1.x)) + ((p2.y - p1.y) * (p2.y - p1.y)));
}

//
void cropImageSet(Mat * image,int size, std::vector<Point> * points, Mat * result){
	int xWidth, yWidth;
	for(int i = 0; i < size; i++){

		//xWidth = distance(points[i][0],points[i][2]);
		//xWidth += (distance(points[i][3],points[i][1])) / 2;

		//yWidth = distance(points[i][0],points[i][3]);
		//yWidth += (distance(points[i][2],points[i][1])) / 2;

		//cropImage(image[i], points[i][0].x , points[i][0].y, xWidth, yWidth, result[i]);
		imshow("cropped",result[i]);
		waitKey(0);
	}
		
}

#pragma endregion CROPPING

// Contains the code to do with the geometric transformations
// I based my answer off the sample code.
#pragma region GEOMETRIC TRANSFORMATION

void transform(Mat imageToTransform, std::vector<Point> srcPoints, std::vector<Point> dstPoints, Mat result){
	Point2f source_points[4], destination_points[4];
	
	source_points[0] = Point2f( srcPoints[0].x, srcPoints[0].y ); // top left
	source_points[1] = Point2f( srcPoints[1].x, srcPoints[1].y ); // top right
	source_points[2] = Point2f( srcPoints[2].x, srcPoints[2].y ); // bottom left
	source_points[3] = Point2f( srcPoints[3].x, srcPoints[3].y ); // bottom right

	destination_points[2] = Point2f( dstPoints[0].x, dstPoints[0].x );
	destination_points[1] = Point2f( dstPoints[1].x, dstPoints[1].x );
	destination_points[0] = Point2f( dstPoints[2].x, dstPoints[2].x );
	destination_points[3] = Point2f( dstPoints[3].x, dstPoints[3].x );

	Mat perspective_matrix( 3, 3, CV_32FC1 );
	perspective_matrix = getPerspectiveTransform( source_points, destination_points );

	warpPerspective( imageToTransform, result, perspective_matrix, result.size() );
	imshow("",imageToTransform);
	imshow("",result);
	waitKey(0);
}

std::vector<Point> getTemplatePoints(std::vector<Point> pointsInImages){
	int minX = 0, maxX = std::numeric_limits<int>::max();
	int minY = 0, maxY = std::numeric_limits<int>::max();
	std::vector<Point> cornerPoints;
	for(int i = 0; i < pointsInImages.size(); i++){
		// max values for X and Y
		if(maxX > pointsInImages[i].x) maxX = pointsInImages[i].x;
		if(maxY > pointsInImages[i].y) maxY = pointsInImages[i].y;

		// min values for X and Y
		if(minX < pointsInImages[i].x) minX = pointsInImages[i].x;
		if(minY < pointsInImages[i].y) minY = pointsInImages[i].y;

	}
	cornerPoints.push_back(Point(minX,minY));// top left
	cornerPoints.push_back(Point(maxX,minY));// top right
	cornerPoints.push_back(Point(minX,maxY));// bottom left
	cornerPoints.push_back(Point(maxX,maxY));// bottom right
	return cornerPoints;
}

std::vector<Point> getTemplateCorners(Mat * templateImages, int size){
	Mat * mask = new Mat[size];;
	Mat * backProjectedImages = new Mat[size];
	getMask(templateImages,size,mask);
	backProjectionAndThreshold(mask,size,backProjectedImages);
	std::vector<Point> temp = getWhiteDotsLocations(backProjectedImages[0]);
	std::vector<Point> result = getTemplatePoints(temp);
	return result;
}
#pragma endregion GEOMETRIC TRANSFORMATION

// Function to run program
int main(int argc, const char** argv){	
	int bookSize = sizeof(books) / sizeof(books[0]);
	int pageSize = sizeof(pages) / sizeof(pages[0]);

	Mat * booksMat = new Mat[bookSize];
	Mat * backProjectionImages = new Mat[bookSize];
	Mat * mask = new Mat[bookSize];
	Mat * croppedImages = new Mat[bookSize];
	Mat * transformation = new Mat[bookSize];

	Mat * resizedPages = new Mat[pageSize];
	Mat * templatePages = new Mat[pageSize];
	Mat * pagesMat = new Mat[pageSize];

	std::vector<Point> * whiteDotsLocation = new std::vector<Point>[bookSize];
	std::vector<Point> templateCorners;

	sampleBluePixel = cv::imread(sampleBlue, -1);
	loadImages(bookLoc, books, bookSize, booksMat);
	loadImages(pageLoc, pages, pageSize, pagesMat);
	templateCorners = getTemplateCorners(pagesMat, pageSize);
	// resize to improve speed
	getMask(booksMat,bookSize,mask);

	// backproject and threshold, then draw circles where the corners are
	backProjectionAndThreshold(mask,bookSize,backProjectionImages);
	drawLocationOfPage(backProjectionImages, mask ,bookSize, whiteDotsLocation);
	
	transform(mask[0], whiteDotsLocation[0],templateCorners,transformation[0]);

	//cropImageSet(originalResizedBooks,bookSize,whiteDotsLocation,croppedImages);

	displayImages("Found Corners",bookSize,booksMat,mask,backProjectionImages);
	//displayImages("Found Corners",bookSize,originalResizedBooks,mask,backProjectionImages);
    return 0;
}