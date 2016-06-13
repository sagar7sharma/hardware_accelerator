#ifndef KAZE_H_
#define KAZE_H_

/******************************************** INCLUDES ********************************************/

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/****************************************** DEFINITIONS *******************************************/

#define DEFAULT_OCTAVE_MAX   3		// Maximum number of octaves
#define DEFAULT_NSUBLEVELS   5		// Maximum number of sublevels per octave
#define EVOLUTION_LEVELS     15		// Toatal number of scale levels
#define DEFAULT_SCALE_OFFSET 1.0f	// Base scale offset (sigma units)
#define KCONTRAST_PERCENTILE 0.60f	// Percentile level for the contrast factor
#define KCONTRAST_NBINS      300	// Number of bins for the contrast factor histogram

#define FALSE   0
#define TRUE    1
#define TAU_MAX 0.25f
#define PI      3.14159265358979323846f

/***************************************** GLOBAL STRUCTS *****************************************/

struct KeyPoints {
	float x_coord;		// X-coordinate of the key-point
	float y_coord;		// Y-coordinate of the key-point
	uint8_t level;		// Level in scale space
	uint8_t scale;		// Diameter of the meaningful keypoint neighborhood
	uint8_t octave;		// Octave (pyramid layer) from which the key-point has been extracted	
	uint8_t sublevel;	// Sublevel of the corresponding octave
	float value;		// The response by which the most strong key-points have been selected
};

/**************************************** GLOBAL VARIABLES ****************************************/

extern struct KeyPoints *key_points;	// Vector of found key-points
extern int16_t *descriptor;             // Vector of Descriptors
extern uint16_t img_cols, img_rows;		// Number of image colums, rows
extern uint32_t img_pixels;				// Number of image pixels

/************************************ FLOATING POINT FUNCTIONS ************************************/

int16_t kaze_init(void);
float Feature_Detection1(const uint8_t *data);
void Festure_Detection2(const float contrast);
uint32_t Feature_Detection3(const float detector_threshold);
void Feature_Description(const uint32_t number_keypoints);
void kaze_exit(void);

/************************************* FIXED POINT FUNCTIONS **************************************/

int16_t kaze_init_fp(void);
uint32_t Feature_Detection1_fp(const uint8_t *data);
void Festure_Detection2_fp(const uint32_t contrast);
uint32_t Feature_Detection3_fp(const int32_t detector_threshold);
uint32_t Feature_Detection_fp(const int32_t detector_threshold);
void Feature_Description_fp(const uint32_t number_keypoints);

/**************************************************************************************************/

#endif // KAZE_H_
