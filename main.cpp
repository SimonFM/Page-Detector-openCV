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


using namespace std;
using namespace cv;

// Location of the images in the project
char * bookLoc = "Media/Books/";
char * pageLoc = "Media/Pages/";

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

// Testing
void loadImages(char * fileLocation, char ** imageFiles, int size, cv::Mat * &images){
	int number_of_images = sizeof(imageFiles)/sizeof(imageFiles[0]);

	images = new cv::Mat[size];

	// This code snippet was taken from your OpenCVExample and it loads the images
	for(int i = 0 ; i < size ; i++){
		string filename(fileLocation);
		filename.append(imageFiles[i]);
		images[i] = cv::imread(filename, -1);

		if (images[i].empty()) {
			cout << "Could not open " << filename << endl;
		}
		else{
			cout << "Opened: " << filename << endl;
		}
	}
}

// Function to run program
int main(int argc, const char** argv){
	int bookSize = sizeof(books) / sizeof(books[0]);
	int pageSize = sizeof(pages) / sizeof(pages[0]);
	int i;
	cv::Mat * booksMat = new cv::Mat[bookSize];
	cv::Mat * pagesMat = new cv::Mat[pageSize];

	loadImages(bookLoc, books, bookSize, booksMat);
	loadImages(pageLoc, pages, pageSize, pagesMat);
	for(i = 0 ; i < bookSize; i++){
		cv::imshow("Bookview",booksMat[i]);
		cv::waitKey(0);
	}

	for(i = 0 ; i < pageSize; i++){
		cv::imshow("Page",pagesMat[i]);
		cv::waitKey(0);
	}

    return 0;
}
