//=============================================================================
//
// kaze_compare.cpp
// Author: Pablo F. Alcantarilla
// Date: 11/12/2012
// Email: pablofdezalc@gmail.com
//
// KAZE Features Copyright 2014, Pablo F. Alcantarilla
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

#include <fstream>
#include <opencv/cvaux.h>
#include <opencv/highgui.h>
#include <opencv2/features2d/features2d.hpp>

extern "C" { 
	#include "kaze.h"
        
}
extern "C" int16_t kaze_init(void);
//float Feature_Detection1(const uint8_t *data);
//void Festure_Detection2(const float contrast);
//uint32_t Feature_Detection3(const float detector_threshold);
//void Feature_Description(const uint32_t number_keypoints);
//void kaze_exit(void);

using namespace std;
using namespace cv;

/****************************************** DEFINITIONS *******************************************/

#define NUMBER_OF_KEYPOINTS  32768		// Maximum Number of Keypoints that can be Stored
#define FIXED_POINT          FALSE		// Use Floating Point or Fixed Point Implementation
#define DETECTOR_THRESHOLD   0.001f		// Detector Response Threshold to Accept Point
#define DETECTOR_THRESHOLD_FP (int32_t)(DETECTOR_THRESHOLD * powf(2, 32))

/**************************************** GLOBAL VARIABLES ****************************************/

// Image Paths
/*
static const char img_path1[]       = "C:/Users/Lester/Dropbox/kaze/images_test/face.jpg";
const int img_num = 14;
static const char* img_path2[img_num] = {
	"C:/Users/Lester/Dropbox/kaze/images_test/face (1).jpg",
	"C:/Users/Lester/Dropbox/kaze/images_test/face (2).jpg",
	"C:/Users/Lester/Dropbox/kaze/images_test/face (3).jpg",
	"C:/Users/Lester/Dropbox/kaze/images_test/face (4).jpg",
	"C:/Users/Lester/Dropbox/kaze/images_test/face (5).jpg",
	"C:/Users/Lester/Dropbox/kaze/images_test/face (6).jpg",
	"C:/Users/Lester/Dropbox/kaze/images_test/face (7).jpg",
	"C:/Users/Lester/Dropbox/kaze/images_test/face (8).jpg",
	"C:/Users/Lester/Dropbox/kaze/images_test/face (9).jpg",
	"C:/Users/Lester/Dropbox/kaze/images_test/face (10).jpg",
	"C:/Users/Lester/Dropbox/kaze/images_test/face (11).jpg",
	"C:/Users/Lester/Dropbox/kaze/images_test/face (12).jpg",
	"C:/Users/Lester/Dropbox/kaze/images_test/face (13).jpg",
	"C:/Users/Lester/Dropbox/kaze/images_test/face (14).jpg"
};*/

static const char img_path1[] = "/home/varun/Desktop/kaze/img1.pgm";
const int img_num = 5;
static const char* img_path2[img_num] = {
	"/home/varun/Desktop/kaze/img2.pgm",
	"/home/varun/Desktop/kaze/img3.pgm",
	"/home/varun/Desktop/kaze/img4.pgm",
	"/home/varun/Desktop/kaze/img5.pgm",
	"/home/varun/Desktop/kaze/img6.pgm",
};
static const char* homography_path[] = {
	"/home/varun/Desktop/kaze/H1to2p",
	"/home/varun/Desktop/kaze/H1to3p",
	"/home/varun/Desktop/kaze/H1to4p",
	"/home/varun/Desktop/kaze/H1to5p",
	"/home/varun/Desktop/kaze/H1to6p",
};


struct KeyPoints *key_points = NULL;	// Vector of Found Key-Points
int16_t *descriptor = NULL;				// Vector of Descriptors
uint16_t img_cols = 320;				// Number of Image Colums
uint16_t img_rows = 240;				// Number of Image Rows
uint32_t img_pixels = 76800;			// Number of Image Pixels

/************************************* FUNCTIONS DECLARATIONS *************************************/

// Print Information for Debugging
void print_debug(int number_keypoints, int16_t *desc, struct KeyPoints *key_points);
// Function for reading the ground truth homography from a txt file
bool read_homography(const std::string& homography_path, cv::Mat& H1toN);
// This function computes the set of inliers given a ground truth homography
void compute_inliers_homography(const std::vector<cv::Point2f>& matches,
	std::vector<cv::Point2f>& inliers, const cv::Mat& H, float min_error);
// This function computes the set of inliers estimating the fundamental matrix
void compute_inliers_ransac(const std::vector<cv::Point2f>& matches,
	std::vector<cv::Point2f>& inliers, float error, bool use_fund);
// This function converts matches to points using nearest neighbor distance ratio matching strategy
void matches2points_nndr(const std::vector<cv::KeyPoint>& train,
	const std::vector<cv::KeyPoint>& query, const std::vector<std::vector<cv::DMatch> >& matches,
	std::vector<cv::Point2f>& pmatches, const float nndr);
// This function draws the list of detected keypoints
void draw_keypoints(cv::Mat& img, const std::vector<cv::KeyPoint>& kpts);
// This function draws the set of the inliers between the two images
void draw_inliers(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& img_com,
	const std::vector<cv::Point2f>& ptpairs, int color);

/***************************************************************************************************
*********************************************** MAIN ***********************************************
***************************************************************************************************/

int main(void) {

	/************************************ KAZE Initialization ************************************/

	// Compute Average Results
	int32_t res_matches = 0;
	int32_t res_inliers = 0;
	double res_ex_time = 0.0;

	uint32_t number_keypoints1 = 0, number_keypoints2 = 0;

	// Allocate Memory for Key-Points
	key_points = new KeyPoints[NUMBER_OF_KEYPOINTS];
	if (key_points == NULL) return -1;

	// Allocate Memory for Descriptors
	descriptor = new int16_t[NUMBER_OF_KEYPOINTS * 64];
	if (descriptor == NULL) return -1;
	
	// Read Input Images
	cv::Mat img1 = imread(img_path1, CV_LOAD_IMAGE_GRAYSCALE);	
	cv::Mat img2 = imread(img_path2[0], CV_LOAD_IMAGE_GRAYSCALE);
	if (img1.data == NULL) return -1;
	if (img2.data == NULL) return -1;
	if (img1.cols != img2.cols || img1.rows != img2.rows) return -1;

	// Set Image Properties
	img_cols = (uint16_t)img1.cols;
	img_rows = (uint16_t)img1.rows;
	img_pixels = (uint32_t)img_cols * (uint32_t)img_rows;	

	// Kaze Initialization
	int16_t init = 0;
	if (FIXED_POINT == TRUE) {
		init = kaze_init_fp();
	} else {
		init = kaze_init();
	}
	if (init == -1) return -1;

	// Repeating Loop for Timing
	const int num_counter = 1;
	for (int counter = 0; counter < num_counter; counter++) {

		/************************************** KAZE 1 Computation **************************************/

		// Kaze Computation for First Image
		const double t1 = (double)(cv::getTickCount());
		if (FIXED_POINT == TRUE) {
			const uint32_t contrast = Feature_Detection1_fp(img1.data);
			Festure_Detection2_fp(contrast);
			number_keypoints1 = Feature_Detection3_fp(DETECTOR_THRESHOLD_FP);
			Feature_Description_fp(number_keypoints1);
		}
		else {
			const float contrast = Feature_Detection1(img1.data);
			Festure_Detection2(contrast);
			number_keypoints1 = Feature_Detection3(DETECTOR_THRESHOLD);
			Feature_Description(number_keypoints1);
		}
		const double t2 = (double)(cv::getTickCount());
		const double tkaze0 = 1000.0*(t2 - t1) / cv::getTickFrequency();

		// Copy Data To OpenCV Space
		cv::Mat desc11 = cv::Mat(number_keypoints1, 64, CV_32FC1);
		for (unsigned int i = 0; i < number_keypoints1 * 64; i++) {
			((float *)(desc11.data))[i] = (float)descriptor[i];
		}
		vector<cv::KeyPoint> key_points11;
		for (unsigned int i = 0; i < number_keypoints1; i++) {
			cv::KeyPoint point1;
			point1.pt.x = key_points[i].x_coord;
			point1.pt.y = key_points[i].y_coord;
			point1.size = key_points[i].scale;
			key_points11.push_back(point1);
		}
		
		res_ex_time += tkaze0;

		for (int img_cnt = 0; img_cnt < img_num; img_cnt++) {

			// Read Input Images
			img2 = imread(img_path2[img_cnt], CV_LOAD_IMAGE_GRAYSCALE);
			if (img2.data == NULL) return -1;
			if (img1.cols != img2.cols || img1.rows != img2.rows) return -1;

			/************************************** KAZE 2 Computation **************************************/

			// Kaze Computation for Second Image
			const double t3 = (double)(cv::getTickCount());
			if (FIXED_POINT == TRUE) {
				const uint32_t contrast = Feature_Detection1_fp(img2.data);
				Festure_Detection2_fp(contrast);
				number_keypoints2 = Feature_Detection3_fp(DETECTOR_THRESHOLD_FP);
				Feature_Description_fp(number_keypoints2);
			}
			else {
				const float contrast = Feature_Detection1(img2.data);
				Festure_Detection2(contrast);
				number_keypoints2 = Feature_Detection3(DETECTOR_THRESHOLD);
				Feature_Description(number_keypoints2);
			}
			const double t4 = (double)(cv::getTickCount());
			const double tkaze1 = 1000.0*(t4 - t3) / cv::getTickFrequency();

			// Copy Data To OpenCV Space
			cv::Mat desc22 = cv::Mat(number_keypoints2, 64, CV_32FC1);
			for (unsigned int i = 0; i < number_keypoints2 * 64; i++) {
				((float *)(desc22.data))[i] = (float)descriptor[i];
			}
			vector<cv::KeyPoint> key_points22;
			for (unsigned int i = 0; i < number_keypoints2; i++) {
				cv::KeyPoint point2;
				point2.pt.x = key_points[i].x_coord;
				point2.pt.y = key_points[i].y_coord;
				point2.size = key_points[i].scale;
				key_points22.push_back(point2);
			}

			/*************************************** KAZE Matching ****************************************/

			// Kaze Matching Time
			const double t5 = (double)(cv::getTickCount());

			// KAZE Matching Variables/Arrays
			cv::Ptr<DescriptorMatcher> matcher_l2 = cv::DescriptorMatcher::create("BruteForce");
			vector<cv::Point2f> matches_kaze, inliers_kaze;
			vector<vector<cv::DMatch> > dmatches_kaze;
			cv::Mat HG;

			// Matching Descriptors
			// matcher_l2->knnMatch(desc11, desc22, dmatches_kaze, 2);
                        // matches2points_nndr(key_points11, key_points22, dmatches_kaze, matches_kaze, 0.8f);
                        vector<Mat> trainDescCollection; 
                        bool crossCheck = false;
                        bool compactResult=false;
                        int knn = 2; 
                        vector<Mat> descriptors = vector<Mat>(1, desc22);
                        dmatches_kaze.clear();
                        trainDescCollection.insert( trainDescCollection.end(), descriptors.begin(), descriptors.end() );
                        CV_Assert( knn > 0 );
                        vector<Mat> masks = vector<Mat>(1,Mat());
                        size_t imageCount = trainDescCollection.size();
                        CV_Assert( masks.size() == imageCount );
                        for( size_t i = 0; i < imageCount; i++ )
                        {
                          if( !masks[i].empty() && !trainDescCollection[i].empty() )
                          {
                            CV_Assert( masks[i].rows == desc11.rows && masks[i].cols == trainDescCollection[i].rows && 
                            masks[i].type() == CV_8UC1 );
                          }
                        }
                        const int IMGIDX_SHIFT = 18;
                        const int IMGIDX_ONE = (1 << IMGIDX_SHIFT);
                        if( desc11.empty() || desc22.empty() )
                        {
                          dmatches_kaze.clear();
                          return -1;
                        }
                        CV_Assert( desc11.type() == trainDescCollection[0].type() );
                        dmatches_kaze.reserve(desc11.rows);
                        int iIdx, imgCount = (int)trainDescCollection.size(), update = 0;
                        int normType = NORM_L2; 
                        int dtype = normType == NORM_HAMMING || normType == NORM_HAMMING2 ||
                        (normType == NORM_L1 && desc11.type() == CV_8U) ? CV_32S : CV_32F;
                        int maxRows = 0;
                        CV_Assert( (int64)imgCount*IMGIDX_ONE < INT_MAX );
                        for( iIdx = 0; iIdx < imgCount; iIdx++ )
                          maxRows = std::max(maxRows, trainDescCollection[iIdx].rows);
                        int m = desc11.rows;
                        Mat dist(m, knn, dtype), nidx(m, knn, CV_32S);
                        //cout << NORM_L2;
                        cout << dmatches_kaze[0][0].distance;
                        dist = Scalar::all(dtype == CV_32S ? (double)INT_MAX : (double)FLT_MAX);
                        nidx = Scalar::all(-1);
                        for( iIdx = 0; iIdx < imgCount; iIdx++ )
                        {
                          CV_Assert( trainDescCollection[iIdx].rows < IMGIDX_ONE );
                          int n = std::min(knn, trainDescCollection[iIdx].rows);
                          Mat dist_i = dist.colRange(0, n), nidx_i = nidx.colRange(0, n);
                          batchDistance(desc11, trainDescCollection[iIdx], dist_i, dtype, nidx_i,
                          normType, knn, masks.empty() ? Mat() : masks[iIdx], update, crossCheck);
                          update += IMGIDX_ONE;
                        }
                        if( dtype == CV_32S )
                        {
                          Mat temp;
                          dist.convertTo(temp, CV_32F);
                          dist = temp;
                        }
                        for( int qIdx = 0; qIdx < desc11.rows; qIdx++ )
                        {
                          const float* distptr = dist.ptr<float>(qIdx);
                          const int* nidxptr = nidx.ptr<int>(qIdx);

                          dmatches_kaze.push_back( vector<DMatch>() );
                          vector<DMatch>& mq = dmatches_kaze.back();
                          mq.reserve(knn);

                            for( int k = 0; k < nidx.cols; k++ )
                            {
                              if( nidxptr[k] < 0 )
                              break;
                              mq.push_back( DMatch(qIdx, nidxptr[k] & (IMGIDX_ONE - 1),nidxptr[k] >> IMGIDX_SHIFT, distptr[k]) );
                            }

                         if( mq.empty() && compactResult )
                           dmatches_kaze.pop_back();
                        }
                        //cout << dist;
			matches2points_nndr(key_points11, key_points22, dmatches_kaze, matches_kaze, 0.8f);
                        

			// Kaze Matching Time
			const double t6 = (double)(cv::getTickCount());
			const double tkaze2 = 1000.0*(t6 - t5) / cv::getTickFrequency();

			/************************************* KAZE Visualization *************************************/

			// Color Images to Visualize Results
			cv::Mat img1_rgb_kaze = cv::Mat(cv::Size(img1.cols, img1.rows), CV_8UC3);
			cv::Mat img2_rgb_kaze = cv::Mat(cv::Size(img2.cols, img2.rows), CV_8UC3);
			cv::Mat img_com_kaze = cv::Mat(cv::Size(img1.cols * 2, img1.rows), CV_8UC3);

			// Read the Homography File	
			bool use_ransac = false;
			if (read_homography(homography_path[img_cnt], HG) == false)
				use_ransac = true;

			// Compute Inliers
			if (use_ransac == false)
				compute_inliers_homography(matches_kaze, inliers_kaze, HG, 2.5f);
			else
				compute_inliers_ransac(matches_kaze, inliers_kaze, 2.5f, false);

			// Compute Kaze Results
			int32_t nmatches_kaze = matches_kaze.size() / 2;
			int32_t ninliers_kaze = inliers_kaze.size() / 2;
			int32_t noutliers_kaze = nmatches_kaze - ninliers_kaze;
			float nratio_kaze = (float)(200 * ninliers_kaze) / (float)(number_keypoints1 + number_keypoints2);
			float ratio_kaze = (nmatches_kaze != 0) ? ((float)(100 * ninliers_kaze) / (float)nmatches_kaze) : (0.f);

			// Prepare the Visualization
			cv::cvtColor(img1, img1_rgb_kaze, cv::COLOR_GRAY2BGR);
			cv::cvtColor(img2, img2_rgb_kaze, cv::COLOR_GRAY2BGR);

			// Draw the List of Detected Key-Points
			draw_keypoints(img1_rgb_kaze, key_points11);
			draw_keypoints(img2_rgb_kaze, key_points22);

			// Create the New Image with a Line Showing the Correspondences
			draw_inliers(img1_rgb_kaze, img2_rgb_kaze, img_com_kaze, inliers_kaze, 2);

			/**************************************** KAZE Results ****************************************/
			
			cout << "KAZE Results" << endl;
			cout << "**************************************" << endl;
			cout << "Number of Keypoints Image 1: " << number_keypoints1 << endl;
			cout << "Number of Keypoints Image 2: " << number_keypoints2 << endl;
			cout << "Number of Matches:  " << nmatches_kaze << endl;
			cout << "Number of Inliers:  " << ninliers_kaze << endl;
			cout << "Number of Outliers: " << noutliers_kaze << endl;
			cout << "Inliers-Matches Ratio:   " << ratio_kaze << endl;
			cout << "Inliers-Keypoints Ratio: " << nratio_kaze << endl;
			cout << "**************************************" << endl;
			cout << "Time for Image1:   " << tkaze0 << endl;
			cout << "Time for Image2:   " << tkaze1 << endl;
			cout << "Time for Matching: " << tkaze2 << endl << endl;
			
			// Show Results
			cv::imshow("KAZE", img_com_kaze);
			waitKey(0);
			
			// Compute Average Results
			res_matches += nmatches_kaze;
			res_inliers += ninliers_kaze;
			res_ex_time += tkaze1;

			// Free Memory
			matcher_l2.release();
			desc22.release();
			HG.release();
			img1_rgb_kaze.release();
			img2_rgb_kaze.release();
			img_com_kaze.release();
		}
		// Free Memory
		desc11.release();
	}

	/**********************************************************************************************/
	
	// Compute Average Results
	float res_i_ratio = (res_matches != 0) ? ((float)(100 * res_inliers) / (float)res_matches) : (0.f);

	cout << "Average KAZE Results" << endl;
	cout << "**************************************" << endl;
	cout << "Number of Matches:       " << res_matches / (5 * num_counter) << endl;
	cout << "Number of Inliers:       " << res_inliers / (5 * num_counter) << endl;
	cout << "Inliers-Keypoints Ratio: " << res_i_ratio << endl;
	cout << "Time for Image:          " << res_ex_time / (6 * num_counter) << endl;

	// Free Memory
	kaze_exit();
	delete[] key_points;
	delete[] descriptor;
	img1.release();
	img2.release();

	// End Program
	destroyAllWindows();
	getchar();
	return 0;
}


/***************************************************************************************************
******************************************** FUNCTIONS *********************************************
***************************************************************************************************/

// Print Information for Debugging
void print_debug(int number_keypoints, int16_t *desc, struct KeyPoints *key_points) {
	float sum_desc = 0.0;
	for (int i = 0; i < number_keypoints * 64; i++) {
		sum_desc += desc[i];
	}
		
	printf("\nSum Descriptor: %f\n", sum_desc);

	float sum_x = 0.0, sum_y = 0.0, sum_value = 0.0;
	uint32_t sum_octave = 0, sum_level = 0, sum_scale = 0, sum_sublevel = 0;
	for (int i = 0; i < number_keypoints; i++) {
		sum_x += key_points[i].x_coord;
		sum_y += key_points[i].y_coord;
		sum_value += key_points[i].value;
		sum_scale += (uint32_t)key_points[i].scale;
		sum_octave += (uint32_t)key_points[i].octave;
		sum_level += (uint32_t)key_points[i].level;
		sum_sublevel += (uint32_t)key_points[i].sublevel;
	}
	printf("sum_x: %f\n", sum_x);
	printf("sum_y: %f\n", sum_y);
	printf("sum_value: %f\n", sum_value);
	printf("sum_esigma: %d\n", sum_scale);
	printf("sum_octave: %d\n", sum_octave);
	printf("sum_level: %d\n", sum_level);
	printf("sum_sublevel: %d\n", sum_sublevel);
}

// Function for reading the ground truth homography from a txt file
// @param homography_file Path for the file that contains the ground truth homography
// @param HG Matrix to store the ground truth homography
bool read_homography(const string& hFile, cv::Mat& H1toN) {

	float h11 = 0.0, h12 = 0.0, h13 = 0.0;
	float h21 = 0.0, h22 = 0.0, h23 = 0.0;
	float h31 = 0.0, h32 = 0.0, h33 = 0.0;
	const int tmp_buf_size = 256;
	char tmp_buf[tmp_buf_size];

	// Allocate memory for the OpenCV matrices
	H1toN = cv::Mat::zeros(3, 3, CV_32FC1);

	string filename(hFile);
	ifstream pf;
	pf.open(filename.c_str(), std::ifstream::in);

	if (!pf.is_open())
		return false;

	pf.getline(tmp_buf, tmp_buf_size);
	sscanf(tmp_buf, "%f %f %f", &h11, &h12, &h13);

	pf.getline(tmp_buf, tmp_buf_size);
	sscanf(tmp_buf, "%f %f %f", &h21, &h22, &h23);

	pf.getline(tmp_buf, tmp_buf_size);
	sscanf(tmp_buf, "%f %f %f", &h31, &h32, &h33);

	pf.close();

	H1toN.at<float>(0, 0) = h11 / h33;
	H1toN.at<float>(0, 1) = h12 / h33;
	H1toN.at<float>(0, 2) = h13 / h33;

	H1toN.at<float>(1, 0) = h21 / h33;
	H1toN.at<float>(1, 1) = h22 / h33;
	H1toN.at<float>(1, 2) = h23 / h33;

	H1toN.at<float>(2, 0) = h31 / h33;
	H1toN.at<float>(2, 1) = h32 / h33;
	H1toN.at<float>(2, 2) = h33 / h33;

	return true;
}

// This function computes the set of inliers given a ground truth homography
//  @param matches Vector of putative matches
//  @param inliers Vector of inliers
//  @param H Ground truth homography matrix 3x3
//  @param min_error The minimum pixelic error to accept an inlier
void compute_inliers_homography(const std::vector<cv::Point2f>& matches,
	std::vector<cv::Point2f>& inliers, const cv::Mat& H, float min_error) {

	float h11 = 0.0, h12 = 0.0, h13 = 0.0;
	float h21 = 0.0, h22 = 0.0, h23 = 0.0;
	float h31 = 0.0, h32 = 0.0, h33 = 0.0;
	float x1 = 0.0, y1 = 0.0;
	float x2 = 0.0, y2 = 0.0;
	float x2m = 0.0, y2m = 0.0;
	float dist = 0.0, s = 0.0;

	h11 = H.at<float>(0, 0);
	h12 = H.at<float>(0, 1);
	h13 = H.at<float>(0, 2);
	h21 = H.at<float>(1, 0);
	h22 = H.at<float>(1, 1);
	h23 = H.at<float>(1, 2);
	h31 = H.at<float>(2, 0);
	h32 = H.at<float>(2, 1);
	h33 = H.at<float>(2, 2);

	inliers.clear();

	for (size_t i = 0; i < matches.size(); i += 2) {
		x1 = matches[i].x;
		y1 = matches[i].y;
		x2 = matches[i + 1].x;
		y2 = matches[i + 1].y;

		s = h31*x1 + h32*y1 + h33;
		x2m = (h11*x1 + h12*y1 + h13) / s;
		y2m = (h21*x1 + h22*y1 + h23) / s;
		dist = sqrt(pow(x2m - x2, 2) + pow(y2m - y2, 2));

		if (dist <= min_error) {
			inliers.push_back(matches[i]);
			inliers.push_back(matches[i + 1]);
		}
	}
}

// This function computes the set of inliers estimating the fundamental matrix
// or a planar homography in a RANSAC procedure
// @param matches Vector of putative matches
// @param inliers Vector of inliers
// @param error The minimum pixelic error to accept an inlier
// @param use_fund Set to true if you want to compute a fundamental matrix
void compute_inliers_ransac(const std::vector<cv::Point2f>& matches,
	std::vector<cv::Point2f>& inliers, float error, bool use_fund) {

	vector<cv::Point2f> points1, points2;
	cv::Mat H = cv::Mat::zeros(3, 3, CV_32F);
	int npoints = matches.size() / 2;
	cv::Mat status = cv::Mat::zeros(npoints, 1, CV_8UC1);

	for (size_t i = 0; i < matches.size(); i += 2) {
		points1.push_back(matches[i]);
		points2.push_back(matches[i + 1]);
	}

	if (npoints > 8) {
		if (use_fund == true)
			H = cv::findFundamentalMat(points1, points2, cv::FM_RANSAC, error, 0.99, status);
		else
			H = cv::findHomography(points1, points2, cv::RANSAC, error, status);

		for (int i = 0; i < npoints; i++) {
			if (status.at<unsigned char>(i) == 1) {
				inliers.push_back(points1[i]);
				inliers.push_back(points2[i]);
			}
		}
	}
}

// This function converts matches to points using nearest neighbor distance ratio matching strategy
// @param train Vector of keypoints from the first image
// @param query Vector of keypoints from the second image
// @param matches Vector of nearest neighbors for each keypoint
// @param pmatches Vector of putative matches
// @param nndr Nearest neighbor distance ratio value
void matches2points_nndr(const std::vector<cv::KeyPoint>& train, 
	const std::vector<cv::KeyPoint>& query, const std::vector<std::vector<cv::DMatch> >& matches,
	std::vector<cv::Point2f>& pmatches, const float nndr) {

	float dist1 = 0.0, dist2 = 0.0;
	for (size_t i = 0; i < matches.size(); i++) {
		cv::DMatch dmatch = matches[i][0];
		dist1 = matches[i][0].distance;
		dist2 = matches[i][1].distance;

		if (dist1 < nndr*dist2) {
			pmatches.push_back(train[dmatch.queryIdx].pt);
			pmatches.push_back(query[dmatch.trainIdx].pt);
		}
	}
}

/// This function draws the list of detected keypoints
void draw_keypoints(cv::Mat& img, const std::vector<cv::KeyPoint>& kpts) {

	int x = 0, y = 0;
	float radius = 0.0;

	for (size_t i = 0; i < kpts.size(); i++) {
		x = (int)(kpts[i].pt.x + .5);
		y = (int)(kpts[i].pt.y + .5);
		radius = kpts[i].size / 2.0f;
		cv::circle(img, cv::Point(x, y), (int)(radius*2.50f), cv::Scalar(0, 255, 0), 1);
		cv::circle(img, cv::Point(x, y), 1, cv::Scalar(0, 0, 255), -1);
	}
}

// This function draws the set of the inliers between the two images
// @param img1 First image
// @param img2 Second image
// @param img_com Image with the inliers
// @param ptpairs Vector of point pairs with the set of inliers
// @param color The color for each method
void draw_inliers(const cv::Mat& img1, const cv::Mat& img2, cv::Mat& img_com,
	const std::vector<cv::Point2f>& ptpairs, int color) {

	int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
	int rows1 = 0, cols1 = 0;
	int rows2 = 0, cols2 = 0;
	float ufactor = 0.0, vfactor = 0.0;

	rows1 = img1.rows;
	cols1 = img1.cols;
	rows2 = img2.rows;
	cols2 = img2.cols;
	ufactor = (float)(cols1) / (float)(cols2);
	vfactor = (float)(rows1) / (float)(rows2);

	// This is in case the input images don't have the same resolution
	cv::Mat img_aux = cv::Mat(cv::Size(img1.cols, img1.rows), CV_8UC3);
	cv::resize(img2, img_aux, cv::Size(img1.cols, img1.rows), 0, 0, cv::INTER_LINEAR);

	for (int i = 0; i < img_com.rows; i++) {
		for (int j = 0; j < img_com.cols; j++) {
			if (j < img1.cols) {
				*(img_com.ptr<unsigned char>(i)+3 * j) = *(img1.ptr<unsigned char>(i)+3 * j);
				*(img_com.ptr<unsigned char>(i)+3 * j + 1) = *(img1.ptr<unsigned char>(i)+3 * j + 1);
				*(img_com.ptr<unsigned char>(i)+3 * j + 2) = *(img1.ptr<unsigned char>(i)+3 * j + 2);
			} else {
				*(img_com.ptr<unsigned char>(i)+3 * j) = *(img2.ptr<unsigned char>(i)+3 * 
					(j - img_aux.cols));
				*(img_com.ptr<unsigned char>(i)+3 * j + 1) = *(img2.ptr<unsigned char>(i)+3 * 
					(j - img_aux.cols) + 1);
				*(img_com.ptr<unsigned char>(i)+3 * j + 2) = *(img2.ptr<unsigned char>(i)+3 * 
					(j - img_aux.cols) + 2);
			}
		}
	}

	for (size_t i = 0; i < ptpairs.size(); i += 2) {
		x1 = (int)(ptpairs[i].x + .5);
		y1 = (int)(ptpairs[i].y + .5);
		x2 = (int)(ptpairs[i + 1].x*ufactor + img1.cols + .5);
		y2 = (int)(ptpairs[i + 1].y*vfactor + .5);
		cv::line(img_com, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 1);
	}
}


/*	float contrast1 = 0.0;
double t1 = 0.0, t2 = 0.0, tkaze0 = 0.0;

const double t11 = (double)(cv::getTickCount());

t1 = (double)(cv::getTickCount());
contrast1 = Feature_Detection1(img1.data);
t2 = (double)(cv::getTickCount());
tkaze0 = 1000.0*(t2 - t1) / cv::getTickFrequency();
cout << "Time for Image1:   " << tkaze0 << endl;

t1 = (double)(cv::getTickCount());
Festure_Detection2(contrast1);
t2 = (double)(cv::getTickCount());
tkaze0 = 1000.0*(t2 - t1) / cv::getTickFrequency();
cout << "Time for Image1:   " << tkaze0 << endl;

t1 = (double)(cv::getTickCount());
number_keypoints1 = Feature_Detection3(DETECTOR_THRESHOLD);
t2 = (double)(cv::getTickCount());
tkaze0 = 1000.0*(t2 - t1) / cv::getTickFrequency();
cout << "Time for Image1:   " << tkaze0 << endl;


const double t22 = (double)(cv::getTickCount());
const double tkaze00 = 1000.0*(t22 - t11) / cv::getTickFrequency();
cout << "Time for Image1:   " << tkaze00 << endl << endl;


t1 = (double)(cv::getTickCount());
Feature_Description(number_keypoints1);
t2 = (double)(cv::getTickCount());
tkaze0 = 1000.0*(t2 - t1) / cv::getTickFrequency();
*/

/*	float contrast2 = 0.0;
double t3 = 0.0, t4 = 0.0, tkaze1 = 0.0;


const double t33 = (double)(cv::getTickCount());

t3 = (double)(cv::getTickCount());
contrast2 = Feature_Detection1(img2.data);
t4 = (double)(cv::getTickCount());
tkaze1 = 1000.0*(t4 - t3) / cv::getTickFrequency();
cout << "Time for Image2:   " << tkaze1 << endl;

t3 = (double)(cv::getTickCount());
Festure_Detection2(contrast2);
t4 = (double)(cv::getTickCount());
tkaze1 = 1000.0*(t4 - t3) / cv::getTickFrequency();
cout << "Time for Image2:   " << tkaze1 << endl;

t3 = (double)(cv::getTickCount());
number_keypoints2 = Feature_Detection3(DETECTOR_THRESHOLD);
t4 = (double)(cv::getTickCount());
tkaze1 = 1000.0*(t4 - t3) / cv::getTickFrequency();
cout << "Time for Image2:   " << tkaze1 << endl;

const double t44 = (double)(cv::getTickCount());
const double tkaze11 = 1000.0*(t44 - t33) / cv::getTickFrequency();
cout << "Time for Image2:   " << tkaze11 << endl << endl;


t3 = (double)(cv::getTickCount());
Feature_Description(number_keypoints2);
t4 = (double)(cv::getTickCount());
tkaze1 = 1000.0*(t4 - t3) / cv::getTickFrequency();
*/
