/*
 * Simon Markham
 * A program that displays an image to the user by detecting a page inside of
 * a given image. The page has a series of blue dots and lines that indicate that
 * this is the page. It only matched 18 out of 25 in the sample set.
 */
#pragma region INCLUDES

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <fstream>
#include <tuple>
#include "Utilities.h"
#include "Histograms.cpp"

using namespace std;
using namespace cv;
#pragma endregion INCLUDES

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
char * sampleBlue = "Media/BlueBookPixels1.png"; // I made my own sample image, It matches 18

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
    images = new Mat[size];

    // This code snippet was taken from your OpenCVExample and it loads the images
    for(int i = 0 ; i < size ; i++){
        string filename(fileLocation);
        filename.append(imageFiles[i]);
        images[i] =  imread(filename, -1);

        if (images[i].empty()) cout << "Could not open " << filename << endl;
        //else cout << "closing: " << filename << endl;
    }
}

// a function that resizes an image with a given factor
void resize(Mat * image,int size, int factor, Mat * result){
    for(int i = 0; i < size; i++)
         resize(image[i],result[i], Size(image[i].cols/factor,image[i].rows/factor));
}

#pragma endregion IMAGE FUNCTIONS

// getWhiteDotsLocations
// getBottomLeftPoint
// getTopRightPoint
// getBottomRightPoint
// getTopLeftPoint
 #pragma region POINTS

// gets the locations of the white dots in an image
 vector<Point> getWhiteDotsLocations(Mat image ){
     vector<Point>  nonZero ;
    findNonZero(image,nonZero);
     vector<Point> result(nonZero.size());
    //for (int i = 0; i < nonZero.size(); i++) result[i] = nonZero[i];

    return nonZero;
}

//gets the bottom left corner in an image
Point getBottomLeftPoint( vector<Point> points){
    Point bottomLeft;
    int size = points.size();
    bottomLeft = points[0];
    for(int i = 0; i < size; i++){
        if( bottomLeft.x > points[i].x && points[i].x > 173)  bottomLeft = points[i];
    }
    return bottomLeft;
}

//gets the bottom right circle in an image
Point getTopRightPoint(Mat image, vector<Point> points){
    Point topRight;
    int size = points.size();
    topRight = points[0];
    for(int i = 0; i < size; i++){
        if( topRight.x < points[i].x && points[i].y > 100) topRight = points[i];
    }
    return topRight;
}

//gets the bottom right circle in an image
Point getBottomRightPoint( vector<Point> points){
    Point bottomRight;
    int size = points.size();
    bottomRight = points[0];
    for(int i = 0; i < size; i++){
        if( bottomRight.y < points[i].y && points[i].y > 100) bottomRight = points[i];
    }
    return bottomRight;
}

//gets the top left circle in an image
Point getTopLeftPoint(  vector<Point> points){
    Point topLeft;
    int size = points.size();
    topLeft = points[0];
    for(int i = 0; i < size; i++){
        if( topLeft.y > points[i].y && points[i].y > 100) topLeft = points[i];
    }
    //cout<< topLeft<< endl;
    return topLeft;
}
#pragma endregion POINTS

// Drawing Lines and Circles
// DisplayingImages
#pragma region DRAWING
// draws lines on an image for a given list of points.
void drawLines(Mat input, vector<Point> points){
    line( input,points[0],points[1],Scalar( 0, 0, 255 ),1,8 );
    line( input,points[1],points[3],Scalar( 0, 0, 255),1,8 );
    line( input,points[2],points[3],Scalar( 0, 0, 255),1,8 );
    line( input,points[2],points[0],Scalar( 0, 0, 255),1,8 );
}

// draws circles on an image for a given list of points.
void drawCircles(Mat input, vector<Point> points, int size){
    circle( input, points[0], 5, Scalar( 0, 0, 0),-1);
    circle( input, points[1], 5, Scalar( 0, 255, 0 ),-1);
    circle( input, points[2], 5, Scalar( 0, 0, 255 ),-1);
    circle( input, points[3], 5, Scalar( 255, 0, 0 ),-1);
}

// A function that displays an array of given images using the imshow function of opencv
// Resizes each image so that it would display nicer on my screen, I then show each image passed
// in to the function
void displayImages(string windowName,int size,Mat * original,Mat * masked, Mat * backProjected, Mat * matchedImages){
    Mat * display = new Mat[size];

    Mat * resizedOriginal = new Mat[size];
    Mat * resizedMasked = new Mat[size];
    Mat * resizedBackProjection = new Mat[size];
    Mat * resizedMatchedImages = new Mat[size];


    Mat temp;
    resize(original,size, THIRD_SIZE, resizedOriginal);
    resize(masked,size, THIRD_SIZE, resizedMasked);
    resize(backProjected,size, THIRD_SIZE, resizedBackProjection);
    resize(matchedImages,size, HALF_SIZE, resizedMatchedImages);

    for(int i = 0 ; i < size; i++){
		// join them all up
        cvtColor(resizedBackProjection[i],temp, CV_GRAY2BGR);
        display[i] = JoinImagesHorizontally(resizedOriginal[i],"Original Image",resizedMasked[i],"Masked Image",2);
        display[i] = JoinImagesHorizontally(display[i],"",temp,"Back Project & closing Image",2);
        display[i] = JoinImagesHorizontally(display[i],"",resizedMatchedImages[i],"Matched Image",2);
		// display them
        imshow(windowName,display[i]);
        waitKey(0);// wait for user input (Enter pressed)
    }
}

//draws the circles and lines on an image
void drawLocationOfPage(Mat * backProjectionImages, Mat * images, int size,  vector<Point> *  whiteDots){
     vector<Point> temp;
    Point * p = new Point[NUM_OF_CORNERS];
    for(int i = 0; i < size; i++){
		// get the points
        temp = getWhiteDotsLocations(backProjectionImages[i]);
        p[0] =  getTopLeftPoint(temp);
        p[1] = getTopRightPoint(backProjectionImages[i],temp);
        p[2] = getBottomLeftPoint(temp);
        p[3] = getBottomRightPoint(temp);

		// gather the points
        for(int j = 0; j < NUM_OF_CORNERS; j++)  whiteDots[i].push_back(p[j]);

        drawCircles(images[i],whiteDots[i],whiteDots[i].size());
        drawLines(images[i],whiteDots[i]);
    }
}
#pragma endregion DRAWING

// backProjectionAndThreshold
// getAndApplyMask
// getRedChannels
// performingDilation
// Some of the code in here was based off the code given to us in the
// OpenCVSample on Blackboard
#pragma region OPERATIONS

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
void getAndApplyMask(Mat * images, int size, Mat * &result){
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


// Contains the code to do with the geometric transformations
// I based my answer off the sample code.
#pragma region GEOMETRIC TRANSFORMATION

// transforms any given image to a given destination.
void transformImage(Mat imageToTransform,  vector<Point> srcPoints,  vector<Point> dstPoints, Mat &result){
    Point2f src[4], dst[4];
	// get the points in the src to transform
    src[0] = Point2f( srcPoints[0].x, srcPoints[0].y ); // top left
    src[1] = Point2f( srcPoints[1].x, srcPoints[1].y ); // top right
    src[2] = Point2f( srcPoints[2].x, srcPoints[2].y ); // bottom left
    src[3] = Point2f( srcPoints[3].x, srcPoints[3].y ); // bottom right

	// this is where the new image will go.
    dst[0] = Point2f( dstPoints[3].x, dstPoints[3].y ); // top left
    dst[1] = Point2f( dstPoints[2].x, dstPoints[2].y ); // top right
    dst[2] = Point2f( dstPoints[1].x, dstPoints[1].y ); // bottom left
    dst[3] = Point2f( dstPoints[0].x, dstPoints[0].y ); // bottom right

    Mat perspective_matrix( 3, 3, CV_32FC1 );
    perspective_matrix = getPerspectiveTransform( src, dst );
    warpPerspective( imageToTransform, result, perspective_matrix, result.size() ); // do the transformation
}

// Apply Transformation to an array of images, given the points and the destinations and store those
// transformed images in result
void transformSetOfImages(Mat * imagesToTransform,  vector<Point>* srcPoints,  vector<Point> dstPoints,int size, Mat * &result){
    for(int i = 0; i < size; i++){
        transformImage(imagesToTransform[i], srcPoints[i],dstPoints,result[i]);
    }

}

// A function that finds the top left, top right, bottom left and the bottom
// right points in a list of points.
 vector<Point> getTemplatePoints( vector<Point> pointsInImages){
    int minX = 0, maxX =  numeric_limits<int>::max();
    int minY = 0, maxY =  numeric_limits<int>::max();
     vector<Point> cornerPoints;
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

// A function that gets the corner points (the blue dots) from each
// template.
 vector<Point> getTemplateCorners(Mat * templateImages, int size){
    Mat * mask = new Mat[size];;
    Mat * backProjectedImages = new Mat[size];
    getAndApplyMask(templateImages,size,mask);
    backProjectionAndThreshold(mask,size,backProjectedImages);
     vector<Point> temp = getWhiteDotsLocations(backProjectedImages[0]);
     vector<Point> result = getTemplatePoints(temp);
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

// Modified from the sample code given to us.
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
        if(max_correlation <  get<0>(maxCorrelations[i]) ) {
            maxIndex = i;
            max_correlation =  get<0>(maxCorrelations[i]);
        }
    }
    return maxIndex;
}


// A function that gets the matched template the for each image
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
		if(results[i] == groundTruths[i]) count++;
		else cout <<"Did not match: Matched Page "<<results[i]<<" and Ground Truth "<<groundTruths[i]<<endl;
		result[i] = templates[results[i]];
	}
	cout <<"Matched a total of: "<< count<<endl;
}
// Function to run program
int main(int argc, const char** argv){
	int bookSize = sizeof(books) / sizeof(books[0]);
	int pageSize = sizeof(pages) / sizeof(pages[0]);
	bookSize = bookSize / 2; // comment this line out if you want the rest of the images
	cout << "Total number of Books: " << bookSize <<endl;
	Mat * booksMat = new Mat[bookSize];
	Mat * backProjectionImages = new Mat[bookSize];
	Mat * masked = new Mat[bookSize];
	Mat * transformedImages = new Mat[bookSize];
	Mat * matchedImages = new Mat[bookSize];

	Mat * thresholded2 = new Mat[bookSize];
	Mat * closing = new Mat[bookSize];
	Mat * backProjected = new Mat[bookSize];
	Mat * croppedImages = new Mat[bookSize];

	Mat * pagesMat = new Mat[pageSize];
	int * results = new int[bookSize];

	// will contain the the white points for all the images
	vector<Point> * whiteDotsLocation = new  vector<Point>[bookSize];
	vector<Point> templateCorners;

	// load in the images
	sampleBluePixel =  imread(sampleBlue, -1);
	cout << "Loading Books..." <<endl;
	loadImages(bookLoc, books, bookSize, booksMat);
	cout << "Loading Templates..." <<endl;
	loadImages(pageLoc, pages, pageSize, pagesMat);

	// This is the functionality
	cout << "Getting template Corners..." <<endl;
	templateCorners = getTemplateCorners(pagesMat, pageSize);
	cout << "Getting and Applying Mask..." <<endl;
	getAndApplyMask(booksMat,bookSize,masked);
	cout << "Doing Back Projection and applying Threshold ..." <<endl;
	backProjectionAndThreshold(masked,bookSize,backProjectionImages);

	cout << "Drawing Corner Locations..." <<endl;
	drawLocationOfPage(backProjectionImages, masked ,bookSize, whiteDotsLocation);
	cout << "Transforming Images..." <<endl;
	transformSetOfImages(booksMat, whiteDotsLocation, templateCorners,bookSize, transformedImages);
	cout << "Doing some Template Matching..." <<endl;
	templateMatchImages(transformedImages,bookSize,pageSize,pagesMat,results);

	cout << "Generating Metrics..." <<endl;
	compareAgainstGroundTruth(transformedImages,pagesMat,results,bookSize,matchedImages);

	// display the result of the process
	cout << "Displaying Images..." <<endl;
	displayImages("Result",bookSize,booksMat,masked,backProjectionImages,matchedImages);
	return 0;
}
