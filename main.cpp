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
#include <tuple>
#include "Utilities.h"
#include "Histograms.cpp"
 
using namespace std;
using namespace cv;
#pragma endregion INCLUDES
 
int image = 0;
#pragma region DEFINES
 
#define NUM_OF_BINS 4
#define NUM_OF_CORNERS  4
#define HALF_SIZE 2
#define THIRD_SIZE 3
 
#pragma endregion DEFINES
 
#pragma region IMAGE LOCATIONS
// Location of the images in the project
char * bookLoc = "Media/Books/";
char * pageLoc = "Media/Pages/";
char * sampleBlue = "Media/BlueBookPixels1.png";
 
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

int groundTruths[] = {0, 1, 2, 3, 4,
					  5, 6, 7, 8, 9,
					  10, 11, 12, 1, 2,
				      4, 3, 6, 8, 7,
					  6, 10, 12, 11, 1};

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
        //else cout << "closing: " << filename << endl;
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
// Resizes each image so that it would display nicer on my screen, I then show each image passed
// in to the function
void displayImages(string windowName,int size,Mat * original,Mat * images, Mat * backProjected, Mat * matchedImages){
    Mat * display = new Mat[size];
 
    Mat * display1 = new Mat[size];
    Mat * display2 = new Mat[size];
    Mat * display3 = new Mat[size];
    Mat * display4 = new Mat[size];
    Mat * display5 = new Mat[size];
 
 
    Mat temp;
    resize(original,size, THIRD_SIZE, display1);
    resize(images,size, THIRD_SIZE, display2);
    resize(backProjected,size, THIRD_SIZE, display3);
    resize(matchedImages,size, HALF_SIZE, display4);
 
    for(int i = 0 ; i < size; i++){
        cvtColor(display3[i],temp, CV_GRAY2BGR);
        display[i] = JoinImagesHorizontally(display1[i],"Original Image",display2[i],"Mask Image",4);
        display[i] = JoinImagesHorizontally(display[i],"",temp,"Back Project & closing Image",4);
        display[i] = JoinImagesHorizontally(display[i],"",display4[i],"Matched Image",4);
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
void greyScaleArray(Mat * &toBeGreyScaled, int size, Mat * &result){
    for(int i = 0; i < size; i++)
        cv::cvtColor(toBeGreyScaled[i],result[i],CV_BGR2GRAY);
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
        threshold(backProjProb, result[i], 15, 255,  CV_THRESH_BINARY );
    }
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
 
 
// gets the Mask of an page so I can only get the page
void performingDilation(Mat * images, int size, Mat * &result){
    for(int i = 0 ;i  < size ; i++)
        morphologyEx(images[i], result[i], MORPH_DILATE, Mat(), Point(-1,-1),1);
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
    std::vector<Point>  nonZero ;
    findNonZero(image,nonZero);
    std::vector<Point> result(nonZero.size());
    //for (int i = 0; i < nonZero.size(); i++) result[i] = nonZero[i];
 
    return nonZero;
}
 
//gets the bottom left corner in an image
Point getBottomLeftPoint(std::vector<Point> points){
    Point bottomLeft;
    int size = points.size();
    bottomLeft = points[0];
    for(int i = 0; i < size; i++){
        if( bottomLeft.x > points[i].x && points[i].x > 173)  bottomLeft = points[i];
    }
    return bottomLeft;
}
 
//gets the bottom right circle in an image
Point getTopRightPoint(Mat image,std::vector<Point> points){
    Point topRight;
    int size = points.size();
    topRight = points[0];
    for(int i = 0; i < size; i++){
        if( topRight.x < points[i].x && points[i].y > 100) topRight = points[i];
    }
    return topRight;
}
 
//gets the bottom right circle in an image
Point getBottomRightPoint(std::vector<Point> points){
    Point bottomRight;
    int size = points.size();
    bottomRight = points[0];
    for(int i = 0; i < size; i++){
        if( bottomRight.y < points[i].y && points[i].y > 100) bottomRight = points[i];
    }
    return bottomRight;
}
 
//gets the top left circle in an image
Point getTopLeftPoint( std::vector<Point> points){
    Point topLeft;
    int size = points.size();
    topLeft = points[0];
    for(int i = 0; i < size; i++){
        if( topLeft.y > points[i].y && points[i].y > 100) topLeft = points[i];
    }
    //cout<< topLeft<< endl;
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
 
void adjustPoints(std::vector<Point> pointsToAdjust){
    std::vector<Point> result;
    Point current;
    pointsToAdjust[0].x = pointsToAdjust[0].x - 5;
    pointsToAdjust[0].y = pointsToAdjust[0].y - 5;
 
    pointsToAdjust[1].x = pointsToAdjust[1].x + 5;
    pointsToAdjust[1].y = pointsToAdjust[1].y - 5;
 
    pointsToAdjust[2].x = pointsToAdjust[2].x + 5;
    pointsToAdjust[2].y = pointsToAdjust[2].y + 5;
 
    pointsToAdjust[3].x = pointsToAdjust[3].x - 5;
    pointsToAdjust[3].y = pointsToAdjust[3].y + 5;
}
 
#pragma endregion POINTS
 
// Contains functions for cropping the images and getting the points
#pragma region CROPPING
//draws the circles and lines on an image
void drawLocationOfPage(Mat * backProjectionImages, Mat * images, int size, std::vector<Point> *  whiteDots){
    std::vector<Point> temp;
    Point * p = new Point[NUM_OF_CORNERS];
    for(int i = 0; i < size; i++){
        temp = getWhiteDotsLocations(backProjectionImages[i]);
        p[0] =  getTopLeftPoint(temp);
        p[1] = getTopRightPoint(backProjectionImages[i],temp);
        p[2] = getBottomLeftPoint(temp);
        p[3] = getBottomRightPoint(temp);
 
        for(int j = 0; j < NUM_OF_CORNERS; j++)  whiteDots[i].push_back(p[j]);
 
        drawCircles(images[i],whiteDots[i],whiteDots[i].size());
        drawLines(images[i],whiteDots[i]);
    }
}
 
// crops an image given a set of points and stored the cropped image in result.
Mat cropImage(Mat image, int x ,int y, int xWidth, int yWidth){
    return image(Rect(x ,y ,xWidth ,yWidth));
}
 
//
double distanceBetween(Point p1, Point p2){
    return sqrt( ((p2.x - p1.x) * (p2.x - p1.x)) + ((p2.y - p1.y) * (p2.y - p1.y)));
}
 
//
void cropImageSet(Mat * imageToCrop,int size, std::vector<Point>  points, Mat * &result){
    int xWidth, yWidth;
    for(int i = 0; i < size; i++){
        /*imshow("i",imageToCrop[i]);
        waitKey(0);*/
        xWidth = (int) distanceBetween(points[2],points[0]);
 
        yWidth = (int) distanceBetween(points[0],points[1]);
 
        result[i] = cropImage(imageToCrop[i],points[3].x , points[3].y,yWidth , xWidth);
    }
}
 
#pragma endregion CROPPING
 
// Contains the code to do with the geometric transformations
// I based my answer off the sample code.
#pragma region GEOMETRIC TRANSFORMATION
 
void transformImage(Mat imageToTransform, std::vector<Point> srcPoints, std::vector<Point> dstPoints, Mat &result){
    Point2f src[4], dst[4];
 
    src[0] = Point2f( srcPoints[0].x, srcPoints[0].y ); // top left
    src[1] = Point2f( srcPoints[1].x, srcPoints[1].y ); // top right
    src[2] = Point2f( srcPoints[2].x, srcPoints[2].y ); // bottom left
    src[3] = Point2f( srcPoints[3].x, srcPoints[3].y ); // bottom right
 
    dst[0] = Point2f( dstPoints[3].x, dstPoints[3].y ); // top left
    dst[1] = Point2f( dstPoints[2].x, dstPoints[2].y ); // top right
    dst[2] = Point2f( dstPoints[1].x, dstPoints[1].y ); // bottom left
    dst[3] = Point2f( dstPoints[0].x, dstPoints[0].y ); // bottom right
 
    Mat perspective_matrix( 3, 3, CV_32FC1 );
    perspective_matrix = getPerspectiveTransform( src, dst );
    warpPerspective( imageToTransform, result, perspective_matrix, result.size() );
}
 
void transformSetOfImages(Mat * imagesToTransform, std::vector<Point>* srcPoints, std::vector<Point> dstPoints,int size, Mat * &result){
    for(int i = 0; i < size; i++){
        transformImage(imagesToTransform[i], srcPoints[i],dstPoints,result[i]);
    }
 
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
 
 
#pragma region TEMPLATE MATCHING
 
// taken from the sample code
void FindLocalMaxima( Mat& input_image, Mat& local_maxima, double threshold_value )
{
    Mat dilated_input_image,thresholded_input_image,thresholded_input_8bit;
    dilate(input_image,dilated_input_image,Mat());
    compare(input_image,dilated_input_image,local_maxima,CMP_EQ);
    threshold( input_image, thresholded_input_image, threshold_value, 255, THRESH_BINARY );
    thresholded_input_image.convertTo( thresholded_input_8bit, CV_8U );
    bitwise_and( local_maxima, thresholded_input_8bit, local_maxima );
}
 
// taken from the sample code
void FindLocalMinima( Mat& input_image, Mat& local_minima, double threshold_value )
{
    Mat eroded_input_image,thresholded_input_image,thresholded_input_8bit;
    erode(input_image,eroded_input_image,Mat());
    compare(input_image,eroded_input_image,local_minima,CMP_EQ);
    threshold( input_image, thresholded_input_image, threshold_value, 255, THRESH_BINARY_INV );
    thresholded_input_image.convertTo( thresholded_input_8bit, CV_8U );
    bitwise_and( local_minima, thresholded_input_8bit, local_minima );
}
 
// taken from the sample code
int templateMatch(Mat full_image, int size ,Mat * templates){
    int i  = 0, maxIndex = 0;
    Mat display_image, correlation_image;
    double min_correlation, max_correlation;
    vector<tuple<double,int>> maxCorrelations;
 
    for(i = 0; i < size; i++){
        int result_columns =  full_image.cols - templates[0].cols + 1;
        int result_rows = full_image.rows - templates[0].rows + 1;
        correlation_image.create( result_columns, result_rows, CV_32FC1 );
        matchTemplate( full_image, templates[i], correlation_image, CV_TM_CCORR_NORMED );
        minMaxLoc( correlation_image, &min_correlation, &max_correlation );
        cout.precision(17);
        maxCorrelations.push_back(make_tuple(max_correlation,i));
 
    }
 
    max_correlation = 0;
    maxIndex = 0;
    for(i = 0 ; i < maxCorrelations.size(); i++){
        if(max_correlation < std::get<0>(maxCorrelations[i]) ) {
            maxIndex = i;
            max_correlation = std::get<0>(maxCorrelations[i]);
        }
    }
 
    //display_image = JoinImagesHorizontally(full_image,"Oringinal",templates[maxIndex],"Template");
    //std::cout << image<<": Max Correlation: " << max_correlation <<std::endl;
 
   // imshow("result",JoinImagesHorizontally(full_image,"transformed Image",templates[maxIndex],"Template"));
  //  waitKey(0);
    //image++;
    return maxIndex;
    /*FindLocalMaxima( correlation_image, matched_template_map, max_correlation * 0.99 );*/
 
}
 

void templateMatchImages(Mat * full_image,int sizeB, int sizeT, Mat * templates, int * result){
 
    for(int i = 0; i < sizeB; i++)
        result[i] = templateMatch(full_image[i],sizeT,templates);
 
}
#pragma endregion TEMPLATE MATCHING

// True Positive = a page is visible and recognised correctly
// False Positive = an incorrectly recognised page, where EITHER no page  was visible OR a different page was visible
// True Negative = no page visible and no page identified
// False Negative = a page is visible but no page was found
void compareAgainstGroundTruth( Mat * images, Mat * templates, int * results, int size, Mat * result){
	int count = 0;

	//for(int i = 0; i < size; i++) cout << results[i]<<endl;
	for(int i = 0; i < size; i++){
		if(results[i] == groundTruths[i]) {
			cout <<"Matched: "<< results[i]<< endl;
			count++;
			
		}
		else cout <<"Did not match"<<results[i]<<" and "<<groundTruths[i]<<endl;
		/*imshow("images",JoinImagesHorizontally(images[groundTruths[i]],"transformed Image",matched[results[i]],"Template"));
		waitKey(0);*/
		//cout <<groundTruths[i] << " and "<< results[i]<<endl;
		result[i] = templates[results[i]];
	}
	cout <<"Matched a total of: "<< count<<endl;
}
// Function to run program
int main(int argc, const char** argv){
    int bookSize = 25;
    int pageSize = sizeof(pages) / sizeof(pages[0]);
 
    Mat * booksMat = new Mat[bookSize];
    Mat * backProjectionImages = new Mat[bookSize];
    Mat * masked = new Mat[bookSize];
    Mat * transformedImages = new Mat[bookSize];
    Mat * matchedImages = new Mat[bookSize];
 
    Mat * greyScaledImages = new Mat[bookSize];
    Mat * thresholded1 = new Mat[bookSize];
    Mat * thresholded2 = new Mat[bookSize];
    Mat * closing = new Mat[bookSize];
    Mat * backProjected = new Mat[bookSize];
    Mat * croppedImages = new Mat[bookSize];
 
 
 
    Mat * resizedPages = new Mat[pageSize];
    Mat * pagesMat = new Mat[pageSize];
	int * results = new int[bookSize];
 
    std::vector<Point> * whiteDotsLocation = new std::vector<Point>[bookSize];
    std::vector<Point> templateCorners, pageCorners;
 
    sampleBluePixel = cv::imread(sampleBlue, -1);
    loadImages(bookLoc, books, bookSize, booksMat);
    loadImages(pageLoc, pages, pageSize, pagesMat);
 
    templateCorners = getTemplateCorners(pagesMat, pageSize);
    pageCorners = getTemplateCorners(pagesMat, pageSize);
 
    getMask(booksMat,bookSize,masked);
    backProjectionAndThreshold(masked,bookSize,backProjectionImages);
    performingDilation(backProjectionImages, bookSize,closing);
    drawLocationOfPage(backProjectionImages, masked ,bookSize, whiteDotsLocation);
    transformSetOfImages(booksMat, whiteDotsLocation, templateCorners,bookSize, transformedImages);
    templateMatchImages(transformedImages,bookSize,pageSize,pagesMat,results);
	compareAgainstGroundTruth(transformedImages,pagesMat,results,bookSize,matchedImages);
    displayImages("Found Corners",bookSize,booksMat,masked,backProjectionImages,matchedImages);
	//displayImages("Cropped",pageSize,pagesMat);
	waitKey(0);
    return 0;
}