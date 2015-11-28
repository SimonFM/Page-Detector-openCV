/*
 * Simon Markham
 * A program that displays an image to the user by detecting a page inside of
 * a given image. The page has a series of blue dots and lines that indicate that
 * this is the page. It only matched 18 out of 25 in the sample set.
 * Code can be found on my github too: https://github.com/SimonFM/Page-Detector-openCV/
 */
#pragma region INCLUDES
 
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <fstream>
#include <tuple>
#include "Headers/Utilities.h"
#include "Histograms.cpp"
#include "Headers/Points.h"
#include "Headers/ImageFunctions.h"
#include "Headers/Drawing.h"
#include "Headers/Operations.h"
#include "Headers/Geometery.h"
#include "Headers/Templates.h"
#include "Headers/Metrics.h"
 
using namespace std;
using namespace cv;
#pragma endregion INCLUDES
 

 
#pragma region IMAGE LOCATIONS
// Location of the images in the project
char * bookLoc = "Media/Books/";
char * pageLoc = "Media/Pages/";
char * sampleBlue = "Media/BlueBookPixels1.png"; // I made my own sample image, It matches 18 
 
 
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
	Mat * cannyImages = new Mat[bookSize];
	Mat * cannyImagesTemplates = new Mat[bookSize];

 
	Mat * pagesMat = new Mat[pageSize];
	int * results = new int[bookSize];
	
	// will contain the the white points for all the images
	vector<Point> * whiteDotsLocation = new  vector<Point>[bookSize];
	vector<Point> templateCorners;
	Mat sampleBluePixel =  imread("Media/BlueBookPixels1.png", -1);
 
	// load in the images
	
	cout << "Loading Books..." <<endl;
	loadImages(bookLoc, books, bookSize, booksMat);
	cout << "Loading Templates..." <<endl;
	loadImages(pageLoc, pages, pageSize, pagesMat);

	// This is the functionality
	cout << "Getting template Corners..." <<endl;
	templateCorners = getTemplateCorners(sampleBluePixel,pagesMat, pageSize);
	cout << "Getting and Applying Mask..." <<endl;
	getAndApplyMask(booksMat,bookSize,masked);
	cout << "Doing Back Projection and applying Threshold ..." <<endl;
	backProjectionAndThreshold(masked,sampleBluePixel,bookSize,backProjectionImages);

	cout << "Drawing Corner Locations..." <<endl;
	drawLocationOfPage(backProjectionImages, masked ,bookSize, whiteDotsLocation);
	cout << "Transforming Images..." <<endl;
	transformSetOfImages(booksMat, whiteDotsLocation, templateCorners,bookSize, transformedImages);

	cout << "Doing some Canny Edge Detection..." <<endl;

	applyCanny(transformedImages,bookSize,cannyImages);
	applyCanny(pagesMat,pageSize,cannyImagesTemplates);
	cout << "Doing some Template Matching..." <<endl;
	templateMatchImages(cannyImages,bookSize,pageSize,cannyImagesTemplates,results);

	cout << "Generating Metrics..." <<endl;
	compareAgainstGroundTruth(transformedImages,pagesMat,results,bookSize,matchedImages);

	// display the result of the process
	cout << "Displaying Images..." <<endl;
	//displayImages("Result",bookSize,booksMat,masked,backProjectionImages,matchedImages);
	displayImages("Result",bookSize,booksMat,matchedImages);
	return 0;
}