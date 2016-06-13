/***************************************************** INCLUDES ******************************************************/

#include "kaze.h"

/***************************************************** ADDRESSES *****************************************************/

//#define ZED_BOARD
#ifdef ZED_BOARD
#define MEM_LxLy02		0x1C000000
#define MEM_LxLy03		0x1C200000
#define MEM_LxLy04		0x1C400000
#define MEM_LxLy05		0x1C600000
#define MEM_LxLy06		0x1C800000
#define MEM_LxLy07		0x1CA00000
#define MEM_LxLy08		0x1CC00000
#define MEM_LxLy09		0x1CE00000
#define MEM_LxLy10		0x1D000000
#define MEM_LxLy11		0x1D200000
#define MEM_LxLy12		0x1D400000
#define MEM_LxLy13		0x1D600000
#define MEM_LxLy14		0x1D800000
#define MEM_Lt			0x1DA00000
#define MEM_Lsmooth		0x1DB00000
#define MEM_Lflow		0x1DC00000
#define MEM_Lstep		0x1DD00000
#define MEM_Ldet02		0x1E000000
#define MEM_Ldet03		0x1E200000
#define MEM_Ldet04		0x1E400000
#define MEM_Ldet05		0x1E600000
#define MEM_Ldet06		0x1E800000
#define MEM_Ldet07		0x1EA00000
#define MEM_Ldet08		0x1EC00000
#define MEM_Ldet09		0x1EE00000
#define MEM_Ldet10		0x1F000000
#define MEM_Ldet11		0x1F200000
#define MEM_Ldet12		0x1F400000
#define MEM_Ldet13		0x1F600000
#define MEM_Ldet14		0x1F800000
#endif

/****************************************************** STRUCTS ******************************************************/

typedef struct {
	uint8_t sigma_size;	// Sigma. For linear diffusion t = sigma^2 / 2
	uint8_t octave;		// Image octave
	uint8_t sublevel;	// Image sublevel in each octave
} Evolution_fp;

typedef struct {
	uint8_t nsteps;		// Number of steps per cycle
	uint32_t* tsteps;	// Vector of FED dynamic time steps (32 bit fraction)
} FED_fp;

typedef struct {
	int16_t Lx;			// First order spatial derivative Lx
	int16_t Ly;			// First order spatial derivative Ly
} Derivative_fp;

typedef struct {
	uint16_t* Lt;		// Evolution image
	uint16_t* Lsmooth;	// Smoothed image
	uint16_t* Lflow;	// Diffusivity image
	int16_t* Lstep;		// Evolution step update	
} ScaleSpace_fp;

/********************************************** GLOBAL VARIABLES/ARRAYS **********************************************/

static Evolution_fp Evolution[EVOLUTION_LEVELS];		// Evolution parameters
static FED_fp FED[EVOLUTION_LEVELS];					// FED parameters

#ifndef ZED_BOARD
	static Derivative_fp* Derivatives[EVOLUTION_LEVELS];	// Lx and Lx Derivatives in scale space
	static ScaleSpace_fp Scalespace[1];						// Non-linear scale space creation
	static int32_t* Determinant[EVOLUTION_LEVELS];			// Detector Responses in scale space	
#else
	static Derivative_fp* Derivatives[] = { NULL, NULL, (Derivative_fp *)MEM_LxLy02, 
		(Derivative_fp *)MEM_LxLy03, (Derivative_fp *)MEM_LxLy04, (Derivative_fp *)MEM_LxLy05,
		(Derivative_fp *)MEM_LxLy06, (Derivative_fp *)MEM_LxLy07, (Derivative_fp *)MEM_LxLy08,
		(Derivative_fp *)MEM_LxLy09, (Derivative_fp *)MEM_LxLy10, (Derivative_fp *)MEM_LxLy11,
		(Derivative_fp *)MEM_LxLy12, (Derivative_fp *)MEM_LxLy13, (Derivative_fp *)MEM_LxLy14 };

	static ScaleSpace_fp Scalespace[] = { { (uint16_t *)MEM_Lt, (uint16_t *)MEM_Lsmooth, 
		(uint16_t *)MEM_Lflow, (int16_t *)MEM_Lstep } };

	static int32_t* Determinant[] = { NULL, NULL, (int32_t *)MEM_Ldet02,
		(int32_t *)MEM_Ldet03, (int32_t *)MEM_Ldet04, (int32_t *)MEM_Ldet05, (int32_t *)MEM_Ldet06,
		(int32_t *)MEM_Ldet07, (int32_t *)MEM_Ldet08, (int32_t *)MEM_Ldet09, (int32_t *)MEM_Ldet10,
		(int32_t *)MEM_Ldet11, (int32_t *)MEM_Ldet12, (int32_t *)MEM_Ldet13, (int32_t *)MEM_Ldet14 };
#endif

/********************************************** FUNCTIONS Initialization *********************************************/

static int16_t fed_tau_by_process_time_fp(const int16_t n, const float T, uint32_t *tau);
static int16_t fed_is_prime_internal_fp(const int16_t number);

/******************************************** FUNCTIONS Feature Detection ********************************************/

static uint32_t Compute_KContrast_fp(const uint16_t *Lt);
static void gaussian5x5_sigma1_init_fp(const uint8_t *Img, uint16_t *Lt);
static void gaussian5x5_sigma1_fp(const uint16_t *Lt, uint16_t *Lsmooth);
static void scharrXY_fp(const uint16_t *Lsmooth, Derivative_fp* Lderiv, const uint8_t scale, const uint16_t scale_N);
static void det_hessian_fp(int32_t *ldet, const Derivative_fp* Lderiv, const uint8_t scale, const uint16_t scale_N);
static void pm_g2_fp(const uint16_t *Lsmooth, uint16_t* Lflow, const uint32_t contrast_square);
static void nld_step_scalar_fp(uint16_t *Lt, const uint16_t *Lflow, int16_t *Lstep, const uint32_t tsteps);


static uint32_t Determinant_Hessian_Parallel_fp(const int32_t detector_threshold);
static uint16_t check_maximum_neighbourhood_fp(const int32_t *img, const int32_t value, 
	                                                   const uint16_t row, const uint16_t col);
static uint32_t Do_Subpixel_Refinement_fp(const uint32_t number_keypoints);

/******************************************* FUNCTIONS Feature Description *******************************************/

static float Compute_Main_Orientation_SURF_fp(const Derivative_fp* Lderiv, const uint8_t scale, const float sig_square, 
	                                          const float xf, const float yf);
static void Get_MSURF_Descriptor_64_fp(const Derivative_fp* Lderiv, const uint8_t scale, const float sig_square, 
	                                   const float xf, const float yf, int16_t* desc, float angle);
static float getAngle_fp(const float x, const float y);
/*
FILE *Lt_fp;
FILE *Ltnext_fp;
FILE *LxLy_fp;
FILE *Ldet_fp;
FILE *Lflow_fp;*/
/**********************************************************************************************************************
******************************************** FUNCTIONS Feature Description ********************************************
**********************************************************************************************************************/

/**
* @brief  This method computes the set of descriptors through the nonlinear scale space
* @param  number_keypoints Vector of keypoints
* @param  vector for descriptors
*/
void Feature_Description_fp(const uint32_t number_keypoints) {

	uint16_t lvl;
	// Loop to cache efficiently use the derivatives
	for (lvl = 3; lvl < EVOLUTION_LEVELS - 1; lvl++) {

		uint32_t i;
		// Compute Descriptor for each key-point
		for (i = 0; i < number_keypoints; i++) {

			const uint8_t level = key_points[i].level;
			if (level != lvl) continue;			
			const float xf = key_points[i].x_coord;
			const float yf = key_points[i].y_coord;
			const uint8_t scale = key_points[i].scale;
			const float sig_square = -1 / (12.5f*(float)scale*(float)scale);
			const Derivative_fp* deriv = Derivatives[level];
			int16_t* desc = &descriptor[i * 64];

			const float angle = Compute_Main_Orientation_SURF_fp(deriv, scale, sig_square, xf, yf);
			Get_MSURF_Descriptor_64_fp(deriv, scale, sig_square, xf, yf, desc, angle);
		}
	}
}

/**
* @brief  This method computes the main orientation for a given keypoint
* @note   The orientation is computed using a similar approach as described in the
*         original SURF method. See Bay et al., Speeded Up Robust Features, ECCV 2006
*/
static float Compute_Main_Orientation_SURF_fp(const Derivative_fp* Lderiv, const uint8_t scale, const float sig_square, 
	                                          const float xf, const float yf) {
	
	float max_prod = 0.0, max_sumY = 0.0, max_sumX = 0.0;
	float resX[109], resY[109], Ang[109];

	uint16_t iy = (uint16_t)(yf + 0.5f) - 5 * (uint16_t)scale;
	const uint16_t xf_round = (uint16_t)(xf + 0.5f);

	// Calculate derivatives responses for points within radius of 6*scale
	uint16_t idx = 0;
	int16_t i;
	for (i = -5; i <= 5; i++) {
		const float y = iy - yf;
		const float yy = y*y;		
		int16_t j;
		for (j = -5; j <= 5; j++) {
			if (i*i + j*j < 36) {
				const uint16_t ix = xf_round + j*(uint16_t)scale;
				const float x = ix - xf;
				const float xx = x*x;
				const float gweight = expf((xx + yy) * sig_square); // gaussian
				const uint32_t ptr = iy*img_cols + ix;
				resX[idx] = gweight*(float)(Lderiv[ptr].Lx);
				resY[idx] = gweight*(float)(Lderiv[ptr].Ly);
				Ang[idx] = getAngle_fp(resX[idx], resY[idx]);
				idx++;
			}
		}
		iy += scale;
	}


	const float a = 6.2831854820251465f; // PI * 2.f;
	const float b = 1.0471975803375244f; // PI / 3.f;
	const float c = 5.2359881401062012f; // 5.f*PI / 3.f;

	// Computing the dominant direction, Loop slides pi/3 window around feature point
	float ang1;
	for (ang1 = 0.0; ang1 < a; ang1 += 0.15f) {
		float sumX = 0.0f, sumY = 0.0f;
		const float ang2 = (ang1 + b > a ? ang1 - c : ang1 + b);
		uint16_t k;
		for (k = 0; k < 109; ++k) { // array size
			// Get angle from the x-axis of the sample point
			const float ang = Ang[k];
			// Determine whether the point is within the window
			if (ang1 < ang2 && ang1 < ang && ang < ang2) {
				sumX += resX[k];
				sumY += resY[k];
			} else if (ang2 < ang1 && ((ang > 0 && ang < ang2) || (ang > ang1 && ang < a))) {
				sumX += resX[k];
				sumY += resY[k];
			}
		}
		// if the vector produced from this window is longer than all
		// previous vectors then this forms the new dominant direction
		const float vec_prod = sumX*sumX + sumY*sumY;
		if (vec_prod > max_prod) {
			max_prod = vec_prod;
			max_sumX = sumX;
			max_sumY = sumY;
		}
	}

	// store largest orientation
	return getAngle_fp(max_sumX, max_sumY);
}

/**
* @brief  This method computes the descriptor of the provided keypoint given the main orientation of the keypoint
* @note   Rectangular grid of 24 s x 24 s. Descriptor Length 64. The descriptor is inspired from Agrawal et al.,
*         CenSurE: Center Surround Extremas for Realtime Feature Detection and Matching, ECCV 2008
*/
static void Get_MSURF_Descriptor_64_fp(const Derivative_fp* Lderiv, const uint8_t scale, const float sig_square, 
	                                   const float xf, const float yf, int16_t* desc, float angle) {

	// Pre-compute the gaussian coefficients
	const float gauss[16] = {
		0.36787945032119751f, 0.57375341653823853f, 0.57375341653823853f, 0.36787945032119751f,
		0.57375341653823853f, 0.89483928680419922f, 0.89483928680419922f, 0.57375341653823853f,
		0.57375341653823853f, 0.89483928680419922f, 0.89483928680419922f, 0.57375341653823853f,
		0.36787945032119751f, 0.57375341653823853f, 0.57375341653823853f, 0.36787945032119751f
	};

	// incremented in each loop
	float descr[64];
	float len = 0.0;
	uint8_t dcount = 0;
	uint8_t gauss_ptr = 0;

	// constant for all loops
	const float si = sinf(angle);
	const float co = cosf(angle);
	const float scale_si = (float)scale*si;
	const float scale_co = (float)scale*co;

	// Calculate descriptor for this interest point,  Area of size 24 s x 24 s
	int16_t i;
	for (i = -7; i < 13; i += 5) {	// pattern size: 12

		const float xs_i = xf + i*scale_co;
		const float ys_i = yf + i*scale_si;

		int16_t j;
		for (j = -7; j < 13; j += 5) {	// pattern size: 12

			float dx = 0.0, dy = 0.0, mdx = 0.0, mdy = 0.0;

			const float xs = xs_i - j*scale_si;
			const float ys = ys_i + j*scale_co;

			int16_t k;
			for (k = i - 5; k < i + 4; k++) {

				const float xf_k = xf + k*scale_co;
				const float yf_k = yf + k*scale_si;

				int16_t l;
				for (l = j - 5; l < j + 4; l++) {

					// Get coords of sample point on the rotated axis
					const float sample_x = xf_k - l*scale_si;
					const float sample_y = yf_k + l*scale_co;

					int16_t y1 = (int16_t)sample_y;
					if (y1 < 0) y1 = 0;
					if (y1 > img_rows - 1) y1 = img_rows - 1;
					int16_t y2 = (int16_t)(sample_y + 1.0f);
					if (y2 < 0) y2 = 0;
					if (y2 > img_rows - 1) y2 = img_rows - 1;

					const uint32_t y1_ptr = (uint32_t)y1*(uint32_t)img_cols;
					const uint32_t y2_ptr = (uint32_t)y2*(uint32_t)img_cols;

					int16_t x1 = (int16_t)sample_x;
					if (x1 < 0) x1 = 0;
					if (x1 > img_cols - 1) x1 = img_cols - 1;
					int16_t x2 = (int16_t)(sample_x + 1.0f);
					if (x2 < 0) x2 = 0;
					if (x2 > img_cols - 1) x2 = img_cols - 1;

					const uint32_t ptr1 = y1_ptr + (uint32_t)x1;
					const uint32_t ptr2 = y1_ptr + (uint32_t)x2;
					const uint32_t ptr3 = y2_ptr + (uint32_t)x1;
					const uint32_t ptr4 = y2_ptr + (uint32_t)x2;

					const float res1x = (float)(Lderiv[ptr1].Lx);
					const float res2x = (float)(Lderiv[ptr2].Lx);
					const float res3x = (float)(Lderiv[ptr3].Lx);
					const float res4x = (float)(Lderiv[ptr4].Lx);
					const float res1y = (float)(Lderiv[ptr1].Ly);
					const float res2y = (float)(Lderiv[ptr2].Ly);
					const float res3y = (float)(Lderiv[ptr3].Ly);
					const float res4y = (float)(Lderiv[ptr4].Ly);

					const float fx = sample_x - x1;
					const float fy = sample_y - y1;
					const float fxn = 1.0f - fx;
					const float fyn = 1.0f - fy;

					const float fxn_fyn = fxn*fyn;
					const float fx_fyn = fx*fyn;
					const float fxn_fy = fxn*fy;
					const float fx_fy = fx*fy;

					const float rx = fxn_fyn*res1x + fx_fyn*res2x + fxn_fy*res3x + fx_fy*res4x;
					const float ry = fxn_fyn*res1y + fx_fyn*res2y + fxn_fy*res3y + fx_fy*res4y;

					// Get the gaussian weighted x and y responses
					const float x = xs - sample_x;
					const float y = ys - sample_y;
					const float gauss_s1 = expf((x*x + y*y) * sig_square);

					// Get the x and y derivatives on the rotated axis
					const float rry = gauss_s1*(rx*co + ry*si);
					const float rrx = gauss_s1*(-rx*si + ry*co);

					// Sum the derivatives to the cumulative descriptor
					dx += rrx;
					dy += rry;
					mdx += fabsf(rrx);
					mdy += fabsf(rry);
				}
			}

			// Add the (4x4) gaussian weighting to the descriptor vector
			float gauss_s2 = gauss[gauss_ptr++];			
			descr[dcount++] = dx*gauss_s2;
			descr[dcount++] = dy*gauss_s2;
			descr[dcount++] = mdx*gauss_s2;
			descr[dcount++] = mdy*gauss_s2;
			len += (dx*dx + dy*dy + mdx*mdx + mdy*mdy)*gauss_s2*gauss_s2;
		}
	}

	// convert to unit vector
	const float Ssqrt = sqrtf(len) / 32768.0f;
	uint16_t k;
	for (k = 0; k < 64; k++) {// discriptor size: 64
		desc[k] = (int16_t)(descr[k] / Ssqrt);
	}
}

/**
* @brief  This function computes the angle from the vector given by (X Y). From 0 to 2*Pi
*/
static float getAngle_fp(const float x, const float y) {
	if (x >= 0 && y >= 0) return atanf(y / x);
	if (x < 0 && y >= 0) return PI - atanf(-y / x);
	if (x < 0 && y < 0) return PI + atanf(y / x);
	if (x >= 0 && y < 0) return 2.0f*PI - atanf(-y / x);
	return 0;
}


/**********************************************************************************************************************
******************************************** FUNCTIONS FEATURE DETECTION 3 ********************************************
**********************************************************************************************************************/

/**
* @brief  This method selects interesting keypoints through the nonlinear scale space
*/
uint32_t Feature_Detection3_fp(const int32_t detector_threshold) {

	// Find scale space extrema
	uint32_t number_keypoints = Determinant_Hessian_Parallel_fp(detector_threshold);

	// Perform some subpixel refinement
	number_keypoints = Do_Subpixel_Refinement_fp(number_keypoints);

	return number_keypoints;
}

/**
* @brief  This method performs the detection of keypoints by using the normalized
*         score of the Hessian determinant through the nonlinear scale space
*/
static uint32_t Determinant_Hessian_Parallel_fp(const int32_t detector_threshold) {

	uint32_t keypoint_cnt1 = 0, keypoint_cnt2 = 0;

	uint16_t lvl;
	for (lvl = 3; lvl < EVOLUTION_LEVELS - 1; lvl++) {
		// skipps borders to check maximum neighbourhood
		uint16_t ix;
		for (ix = 2; ix < img_rows - 2; ix++) {
			const int32_t* const __restrict ldet_m = &Determinant[lvl][ix*img_cols];
			uint16_t jx;
			for (jx = 2; jx < img_cols - 2; jx++) {
				const int32_t value = ldet_m[jx];
				// Filter the points with the detector threshold
				// Check the same, lower and upper scale
				if (value > detector_threshold) {
					if (check_maximum_neighbourhood_fp(Determinant[lvl], value, ix, jx) &&
						check_maximum_neighbourhood_fp(Determinant[lvl - 1], value, ix, jx) &&
						check_maximum_neighbourhood_fp(Determinant[lvl + 1], value, ix, jx)) {

						// Add the point of interest!!  scale will be added later
						key_points[keypoint_cnt1].x_coord = (float)jx;
						key_points[keypoint_cnt1].y_coord = (float)ix;
						key_points[keypoint_cnt1].value = (float)(abs(value));
						key_points[keypoint_cnt1].octave = Evolution[lvl].octave;
						key_points[keypoint_cnt1].level = (uint8_t)lvl;
						key_points[keypoint_cnt1].sublevel = Evolution[lvl].sublevel;
						keypoint_cnt1++;

						// Skip the next two, since they can't be maxima
						jx += 2;
					}
				}
			}
		}
	}

	// Now fill the vector of keypoints!!!
	uint32_t i;
	for (i = 0; i < keypoint_cnt1; i++) {

		uint32_t id_repeated = 0;
		uint8_t is_extremum = TRUE;
		uint8_t is_repeated = FALSE;
		const uint8_t level = key_points[i].level;
		const int32_t scale = (int32_t)(Evolution[level].sigma_size);
		const int32_t scale_square = scale*scale;

		// scip scale size of two, since this was covered in last search
		if (scale > 2) {
			// Check in case we have the same point as maxima in surrounding evolution levels
			uint32_t ik;
			for (ik = 0; ik < keypoint_cnt2; ik++) {
				uint8_t lvl = key_points[ik].level;
				if (lvl == level || lvl == level + 1 || lvl == level - 1) {
					const int32_t dist_x = (int32_t)(key_points[i].x_coord) - (int32_t)(key_points[ik].x_coord);
					const int32_t dist_y = (int32_t)(key_points[i].y_coord) - (int32_t)(key_points[ik].y_coord);
					const int32_t dist = dist_x * dist_x + dist_y * dist_y;
					if (dist < scale_square) {
						if (key_points[i].value > key_points[ik].value) {
							id_repeated = ik;
							is_repeated = TRUE;
						} else {
							is_extremum = FALSE;
						}
						break;
					}
				}
			}
		}
		if (is_extremum == TRUE) {
			if (is_repeated == FALSE) {
				// store new key-point
				key_points[keypoint_cnt2] = key_points[i];
				keypoint_cnt2++;
			} else {
				// overwrite old key-point
				key_points[id_repeated] = key_points[i];
			}
		}
	}
	return keypoint_cnt2;
}

/**
* @brief  This function checks if a given pixel is a maximum in a local neighbourhood
* @param  img Input image where we will perform the maximum search
* @param  value Response value at (x,y) position
* @param  row Image row coordinate
* @param  col Image column coordinate
* @return 1->is maximum, 0->otherwise
*/
static uint16_t check_maximum_neighbourhood_fp(const int32_t *img, const int32_t value, 
	                                                   const uint16_t row, const uint16_t col) {

	const int32_t* const __restrict img_m = &img[row*img_cols];
	const int32_t* const __restrict img_l = img_m - img_cols;
	const int32_t* const __restrict img_h = img_m + img_cols;
	const uint32_t col_l = col - 1;
	const uint32_t col_h = col + 1;

	if (img_l[col - img_cols] > (value)) return FALSE;
	if (img_l[col_l] > (value)) return FALSE;
	if (img_l[col] > (value)) return FALSE;
	if (img_l[col_h] > (value)) return FALSE;
	if (img_m[col - 2] > (value)) return FALSE;
	if (img_m[col_l] > (value)) return FALSE;
	if (img_m[col] > (value)) return FALSE;
	if (img_m[col_h] > (value)) return FALSE;
	if (img_m[col + 2] > (value)) return FALSE;
	if (img_h[col_l] > (value)) return FALSE;
	if (img_h[col] > (value)) return FALSE;
	if (img_h[col_h] > (value)) return FALSE;
	if (img_h[col + img_cols] > (value)) return FALSE;

	return TRUE;
}

/**
* @brief  This method performs subpixel refinement of the detected keypoints
* @param  number_keypoints total number of keypoints bevor function
* @return number_keypoints total number of keypoints after function
*/
static uint32_t Do_Subpixel_Refinement_fp(const uint32_t number_keypoints) {
	uint32_t keypoint_cnt = 0;

	uint32_t cnt;
	for (cnt = 0; cnt < number_keypoints; cnt++) {
		const int32_t N = (int32_t)img_cols;
		const int32_t ptr = (int32_t)(key_points[cnt].y_coord)*N + (int32_t)(key_points[cnt].x_coord);

		const int32_t* const __restrict Ldet_m = &Determinant[key_points[cnt].level][ptr];
		const int32_t* const __restrict Ldet_h = &Determinant[key_points[cnt].level + 1][ptr];
		const int32_t* const __restrict Ldet_l = &Determinant[key_points[cnt].level - 1][ptr];

		const int32_t Ldet_lvl_1h = Ldet_m[1];
		const int32_t Ldet_lvl_1l = Ldet_m[-1];
		const int32_t Ldet_lvl_Nh = Ldet_m[N];
		const int32_t Ldet_lvl_Nl = Ldet_m[-N];
		const int32_t Ldet_lvl = *Ldet_m;
		const int32_t Ldet_lvlh = *Ldet_h;
		const int32_t Ldet_lvll = *Ldet_l;

		// Compute the gradient
		const int32_t Dx = (Ldet_lvl_1h - Ldet_lvl_1l) >> 1;
		const int32_t Dy = (Ldet_lvl_Nh - Ldet_lvl_Nl) >> 1;
		const int32_t Ds = (Ldet_lvlh - Ldet_lvll) >> 1;

		// Compute the Hessian
		const int32_t Dxx = Ldet_lvl_1h + Ldet_lvl_1l - (Ldet_lvl << 1);
		const int32_t Dyy = Ldet_lvl_Nh + Ldet_lvl_Nl - (Ldet_lvl << 1);
		const int32_t Dss = Ldet_lvlh + Ldet_lvll - (Ldet_lvl << 1);
		const int32_t Dxy = (Ldet_m[N + 1] + Ldet_m[-N - 1] - Ldet_m[-N + 1] - Ldet_m[N - 1]) >> 2;
		const int32_t Dxs = (Ldet_h[1] + Ldet_l[-1] - Ldet_h[-1] - Ldet_l[1]) >> 2;
		const int32_t Dys = (Ldet_h[N] + Ldet_l[-N] - Ldet_h[-N] - Ldet_l[N]) >> 2;

		const float fDx = ((float)Dx);
		const float fDy = ((float)Dy);
		const float fDs = ((float)Ds);
		const float fDxx = ((float)Dxx);
		const float fDyy = ((float)Dyy);
		const float fDss = ((float)Dss);
		const float fDxy = ((float)Dxy);
		const float fDxs = ((float)Dxs);
		const float fDys = ((float)Dys);

		// build and solve the linear system with gaussian elimination
		float A[3][4] = { { fDxx, fDxy, fDxs, -fDx }, { fDxy, fDyy, fDys, -fDy }, { fDxs, fDys, fDss, -fDs } };
		float b[3], sum, c;

		// loop for the generation of upper triangular matrix 
		c = A[1][0] / A[0][0];
		A[1][0] = A[1][0] - c*A[0][0];
		A[1][1] = A[1][1] - c*A[0][1];
		A[1][2] = A[1][2] - c*A[0][2];
		A[1][3] = A[1][3] - c*A[0][3];
		c = A[2][0] / A[0][0];
		A[2][0] = A[2][0] - c*A[0][0];
		A[2][1] = A[2][1] - c*A[0][1];
		A[2][2] = A[2][2] - c*A[0][2];
		A[2][3] = A[2][3] - c*A[0][3];
		c = A[2][1] / A[1][1];
		A[2][0] = A[2][0] - c*A[1][0];
		A[2][1] = A[2][1] - c*A[1][1];
		A[2][2] = A[2][2] - c*A[1][2];
		A[2][3] = A[2][3] - c*A[1][3];

		// this loop is for backward substitution
		b[2] = A[2][3] / A[2][2];
		sum = A[1][2] * b[2];
		b[1] = (A[1][3] - sum) / A[1][1];
		sum = A[0][1] * b[1];
		sum += A[0][2] * b[2];
		b[0] = (A[0][3] - sum) / A[0][0];

		// do subpixel refinement
		if (fabsf(b[0]) <= 1.0f && fabsf(b[1]) <= 1.0f && fabsf(b[2]) <= 1.0f) {
			key_points[cnt].x_coord += b[0];
			key_points[cnt].y_coord += b[1];
			const float dsc = (float)key_points[cnt].octave + ((float)key_points[cnt].sublevel + b[2]) / 
				             ((float)DEFAULT_NSUBLEVELS);
			key_points[cnt].scale = (uint8_t)(DEFAULT_SCALE_OFFSET*powf(2.0f, dsc) + 0.5f);

			// Check that the point is under the image limits for the surf main orientation computation
			const int32_t scaleN = (int32_t)(5 * (float)key_points[cnt].scale);
			const int32_t x = (int32_t)(key_points[cnt].x_coord);
			const int32_t y = (int32_t)(key_points[cnt].y_coord);
			if ((x - scaleN) < 0 || (x + scaleN + 1) >= img_cols ||
				(y - scaleN) < 0 || (y + scaleN + 1) >= img_rows) {
			} else {
				// Save refined and not deleted keypoints
				key_points[keypoint_cnt] = key_points[cnt];
				keypoint_cnt++;
			}
		}
	}
	return keypoint_cnt;
}


/**********************************************************************************************************************
******************************************** FUNCTIONS FEATURE DETECTION 2 ********************************************
**********************************************************************************************************************/

/**
* @brief  This method creates the nonlinear scale space for a given image
* @param  data Input image for which the nonlinear scale space needs to be created
*/
void Festure_Detection2_fp(const uint32_t contrast) {

/*	Lt_fp      = fopen("c:\\Lt.txt", "w");
	Ltnext_fp = fopen("c:\\Ltnext.txt", "w");
	LxLy_fp = fopen("c:\\LxLy.txt", "w");
	Ldet_fp = fopen("c:\\Ldet.txt", "w");
	Lflow_fp = fopen("c:\\Lflow.txt", "w");
*/	
	/*********************************************** TEvolution 1..N-2 ***********************************************/

	uint16_t lvl;
	for (lvl = 2; lvl < EVOLUTION_LEVELS - 1; lvl++) {

/*		for (int xx = 0; xx < img_pixels; xx++) {
			for (int i = 15; i >= 0; i--)
				fprintf(Lt_fp, "%d", (Scalespace[0].Lt[xx] >> i) % 2);
			fprintf(Lt_fp, "\n");
		}
*/
		// compute the gaussian smoothing
		gaussian5x5_sigma1_fp(Scalespace[0].Lt, Scalespace[0].Lsmooth);

		// computes the feature detector response for the nonlinear scale space
		// We use the Hessian determinant as feature detector
		const uint8_t scale = Evolution[lvl].sigma_size;
		const uint16_t scale_N = (uint16_t)scale * img_cols;
		scharrXY_fp(Scalespace[0].Lsmooth, Derivatives[lvl], scale, scale_N);
		det_hessian_fp(Determinant[lvl], Derivatives[lvl], scale, scale_N);

		// Compute the conductivity equation and Gaussian derivatives Lx and Ly
		pm_g2_fp(Scalespace[0].Lsmooth, Scalespace[0].Lflow, contrast);

		// Perform FED n inner steps
		uint16_t j;
		for (j = 0; j < FED[lvl].nsteps; j++) {
			nld_step_scalar_fp(Scalespace[0].Lt, Scalespace[0].Lflow, Scalespace[0].Lstep, FED[lvl].tsteps[j]);
		}

/*		for (int xx = 0; xx < img_pixels; xx++) {
			for (int i = 15; i >= 0; i--)
				fprintf(Ltnext_fp, "%d", (Scalespace[0].Lt[xx] >> i) % 2);
			fprintf(Ltnext_fp, "\n");
		}
*/	}

	/************************************************ TEvolution N-1 *************************************************/

/*	for (int xx = 0; xx < img_pixels; xx++) {
		for (int i = 15; i >= 0; i--)
			fprintf(Lt_fp, "%d", (Scalespace[0].Lt[xx] >> i) % 2);
		fprintf(Lt_fp, "\n");
	}*/

	// compute the gaussian smoothing
	gaussian5x5_sigma1_fp(Scalespace[0].Lt, Scalespace[0].Lsmooth);

	// computes the feature detector response for the nonlinear scale space
	// We use the Hessian determinant as feature detector
	const uint8_t scale = Evolution[EVOLUTION_LEVELS - 1].sigma_size;
	const uint16_t scale_N = (uint16_t)scale * img_cols;
	scharrXY_fp(Scalespace[0].Lsmooth, Derivatives[EVOLUTION_LEVELS - 1], scale, scale_N);
	det_hessian_fp(Determinant[EVOLUTION_LEVELS - 1], Derivatives[EVOLUTION_LEVELS - 1], scale, scale_N);

/*	fclose(Lt_fp);
	fclose(Ltnext_fp);
	fclose(LxLy_fp);
	fclose(Ldet_fp); 
	fclose(Lflow_fp);*/
}

/**
* @brief This function smoothes an image with a Gaussian kernel
* @param src Input image
* @param dst Output image
*/
static void gaussian5x5_sigma1_fp(const uint16_t *Lt, uint16_t *Lsmooth) {

	const uint32_t borderY = img_pixels - (uint32_t)img_cols;	// 19|0
	uint32_t yym;
	for (yym = 0; yym < img_pixels; yym += img_cols) {			// 19|0

		// Y-AXIS BORDER
		int32_t yl1 = (int32_t)yym - (int32_t)img_cols;			// 20|0
		if (yl1 < 0) yl1 = 0;
		int32_t yl2 = (int32_t)yym - (int32_t)(img_cols << 1);	// 20|0
		if (yl2 < 0) yl2 = 0;
		uint32_t yh1 = yym + (uint32_t)img_cols;				// 19|0
		if (yh1 == img_pixels) yh1 = borderY;
		uint32_t yh2 = yym + (img_cols << 1);					// 19|0
		if (yh2 >= img_pixels) yh2 = borderY;

		// Y-AXIS POINTER
		const uint16_t* const __restrict src_mm = &Lt[yym];
		const uint16_t* const __restrict src_h1 = &Lt[yh1];
		const uint16_t* const __restrict src_h2 = &Lt[yh2];
		const uint16_t* const __restrict src_l1 = &Lt[yl1];
		const uint16_t* const __restrict src_l2 = &Lt[yl2];
		uint16_t* const __restrict dst_m = &Lsmooth[yym];

		// LEFT BORDER
		// 0|16 + 0|16 .... (unsigned)
		uint32_t temp0 = (((uint64_t)0x6149EA * (uint64_t)(((uint32_t)src_l2[0] + (uint32_t)src_h2[0] + (uint32_t)src_l2[2] + (uint32_t)src_h2[2]) >> 1)) + 0x20000000) >> 30;	// 0x6149EAE9 // 24 x 18
		uint32_t temp1 = (((uint64_t)0x6D0125 * (uint64_t)(((uint32_t)src_l1[2] + (uint32_t)src_h1[2] + (uint32_t)src_l2[1] + (uint32_t)src_h2[1]) >> 1)) + 0x08000000) >> 28;	// 0x6D01250A // 24 x 18
		uint32_t temp2 = (((uint64_t)0x6D0125 * (uint64_t)(((uint32_t)src_l1[0] + (uint32_t)src_h1[0] + (uint32_t)src_l2[0] + (uint32_t)src_h2[0]) >> 1)) + 0x08000000) >> 28;	// 0x6D01250A // 24 x 18
		uint32_t temp3 = (((uint64_t)0x59DBE7 * (uint64_t)(((uint32_t)src_l2[0] + (uint32_t)src_h2[0] + (uint32_t)src_mm[0] + (uint32_t)src_mm[2]) >> 1)) + 0x04000000) >> 27;	// 0x59DBE71E // 24 x 18
		uint32_t temp4 = (((uint64_t)0x7A218B * (uint64_t)(((uint32_t)src_l1[0] + (uint32_t)src_h1[0] + (uint32_t)src_l1[1] + (uint32_t)src_h1[1]) >> 1)) + 0x02000000) >> 26;	// 0x7A218BA5 // 24 x 18
		uint32_t temp5 = (((uint64_t)0x64AE15 * (uint64_t)(((uint32_t)src_l1[0] + (uint32_t)src_h1[0] + (uint32_t)src_mm[0] + (uint32_t)src_mm[1]) >> 1)) + 0x01000000) >> 25;	// 0x64AE16BF // 24 x 18
		uint32_t temp6 = (((uint64_t)0x52FF24 * (uint64_t)((uint32_t)src_mm[0])) + 0x01000000) >> 25;																			// 0x52FF241B // 24 x 16
		dst_m[0] = (uint16_t)(temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6);
		temp0 = (((uint64_t)0x6149EA * (uint64_t)(((uint32_t)src_l2[0] + (uint32_t)src_h2[0] + (uint32_t)src_l2[3] + (uint32_t)src_h2[3]) >> 1)) + 0x20000000) >> 30;	// 0x6149EAE9 // 24 x 18
		temp1 = (((uint64_t)0x6D0125 * (uint64_t)(((uint32_t)src_l1[3] + (uint32_t)src_h1[3] + (uint32_t)src_l2[2] + (uint32_t)src_h2[2]) >> 1)) + 0x08000000) >> 28;	// 0x6D01250A // 24 x 18
		temp2 = (((uint64_t)0x6D0125 * (uint64_t)(((uint32_t)src_l1[0] + (uint32_t)src_h1[0] + (uint32_t)src_l2[0] + (uint32_t)src_h2[0]) >> 1)) + 0x08000000) >> 28;	// 0x6D01250A // 24 x 18
		temp3 = (((uint64_t)0x59DBE7 * (uint64_t)(((uint32_t)src_l2[1] + (uint32_t)src_h2[1] + (uint32_t)src_mm[0] + (uint32_t)src_mm[3]) >> 1)) + 0x04000000) >> 27;	// 0x59DBE71E // 24 x 18
		temp4 = (((uint64_t)0x7A218B * (uint64_t)(((uint32_t)src_l1[0] + (uint32_t)src_h1[0] + (uint32_t)src_l1[2] + (uint32_t)src_h1[2]) >> 1)) + 0x02000000) >> 26;	// 0x7A218BA5 // 24 x 18
		temp5 = (((uint64_t)0x64AE15 * (uint64_t)(((uint32_t)src_l1[1] + (uint32_t)src_h1[1] + (uint32_t)src_mm[0] + (uint32_t)src_mm[2]) >> 1)) + 0x01000000) >> 25;	// 0x64AE16BF // 24 x 18
		temp6 = (((uint64_t)0x52FF24 * (uint64_t)((uint32_t)src_mm[1])) + 0x01000000) >> 25;																			// 0x52FF241B // 24 x 16
		dst_m[1] = (uint16_t)(temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6);

		// MIDDLE
		uint16_t xxm;
		for (xxm = 2; xxm < img_cols - 2; xxm++) {	// 10|0
			const uint16_t xl1 = xxm - 1;
			const uint16_t xl2 = xxm - 2;
			const uint16_t xh1 = xxm + 1;
			const uint16_t xh2 = xxm + 2;
			// 0|16 + 0|16 .... (unsigned)
			temp0 = (((uint64_t)0x6149EA * (uint64_t)(((uint32_t)src_l2[xl2] + (uint32_t)src_h2[xl2] + (uint32_t)src_l2[xh2] + (uint32_t)src_h2[xh2]) >> 1)) + 0x20000000) >> 30;	// 0x6149EAE9 // 24 x 18
			temp1 = (((uint64_t)0x6D0125 * (uint64_t)(((uint32_t)src_l1[xh2] + (uint32_t)src_h1[xh2] + (uint32_t)src_l2[xh1] + (uint32_t)src_h2[xh1]) >> 1)) + 0x08000000) >> 28;	// 0x6D01250A // 24 x 18
			temp2 = (((uint64_t)0x6D0125 * (uint64_t)(((uint32_t)src_l1[xl2] + (uint32_t)src_h1[xl2] + (uint32_t)src_l2[xl1] + (uint32_t)src_h2[xl1]) >> 1)) + 0x08000000) >> 28;	// 0x6D01250A // 24 x 18
			temp3 = (((uint64_t)0x59DBE7 * (uint64_t)(((uint32_t)src_l2[xxm] + (uint32_t)src_h2[xxm] + (uint32_t)src_mm[xl2] + (uint32_t)src_mm[xh2]) >> 1)) + 0x04000000) >> 27;	// 0x59DBE71E // 24 x 18
			temp4 = (((uint64_t)0x7A218B * (uint64_t)(((uint32_t)src_l1[xl1] + (uint32_t)src_h1[xl1] + (uint32_t)src_l1[xh1] + (uint32_t)src_h1[xh1]) >> 1)) + 0x02000000) >> 26;	// 0x7A218BA5 // 24 x 18
			temp5 = (((uint64_t)0x64AE15 * (uint64_t)(((uint32_t)src_l1[xxm] + (uint32_t)src_h1[xxm] + (uint32_t)src_mm[xl1] + (uint32_t)src_mm[xh1]) >> 1)) + 0x01000000) >> 25;	// 0x64AE16BF // 24 x 18
			temp6 = (((uint64_t)0x52FF24 * (uint64_t)((uint32_t)src_mm[xxm])) + 0x01000000) >> 25;																					// 0x52FF241B // 24 x 16
			dst_m[xxm] = (uint16_t)(temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6);
		}

		// RIGHT Border
		const uint16_t xxx = img_cols - 2;
		const uint16_t xl1 = img_cols - 3;
		const uint16_t xl2 = img_cols - 4;
		const uint16_t xh1 = img_cols - 1;
		// 0|16 + 0|16 .... (unsigned)
		temp0 = (((uint64_t)0x6149EA * (uint64_t)(((uint32_t)src_l2[xl2] + (uint32_t)src_h2[xl2] + (uint32_t)src_l2[xh1] + (uint32_t)src_h2[xh1]) >> 1)) + 0x20000000) >> 30;	// 0x6149EAE9 // 24 x 18
		temp1 = (((uint64_t)0x6D0125 * (uint64_t)(((uint32_t)src_l1[xh1] + (uint32_t)src_h1[xh1] + (uint32_t)src_l2[xh1] + (uint32_t)src_h2[xh1]) >> 1)) + 0x08000000) >> 28;	// 0x6D01250A // 24 x 18
		temp2 = (((uint64_t)0x6D0125 * (uint64_t)(((uint32_t)src_l1[xl2] + (uint32_t)src_h1[xl2] + (uint32_t)src_l2[xl1] + (uint32_t)src_h2[xl1]) >> 1)) + 0x08000000) >> 28;	// 0x6D01250A // 24 x 18
		temp3 = (((uint64_t)0x59DBE7 * (uint64_t)(((uint32_t)src_l2[xxx] + (uint32_t)src_h2[xxx] + (uint32_t)src_mm[xl2] + (uint32_t)src_mm[xh1]) >> 1)) + 0x04000000) >> 27;	// 0x59DBE71E // 24 x 18
		temp4 = (((uint64_t)0x7A218B * (uint64_t)(((uint32_t)src_l1[xl1] + (uint32_t)src_h1[xl1] + (uint32_t)src_l1[xh1] + (uint32_t)src_h1[xh1]) >> 1)) + 0x02000000) >> 26;	// 0x7A218BA5 // 24 x 18
		temp5 = (((uint64_t)0x64AE15 * (uint64_t)(((uint32_t)src_l1[xxx] + (uint32_t)src_h1[xxx] + (uint32_t)src_mm[xl1] + (uint32_t)src_mm[xh1]) >> 1)) + 0x01000000) >> 25;	// 0x64AE16BF // 24 x 18
		temp6 = (((uint64_t)0x52FF24 * (uint64_t)((uint32_t)src_mm[xxx])) + 0x01000000) >> 25;																					// 0x52FF241B // 24 x 16
		dst_m[xxx] = (uint16_t)(temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6);
		temp0 = (((uint64_t)0x6149EA * (uint64_t)(((uint32_t)src_l2[xl1] + (uint32_t)src_h2[xl1] + (uint32_t)src_l2[xh1] + (uint32_t)src_h2[xh1]) >> 1)) + 0x20000000) >> 30;	// 0x6149EAE9 // 24 x 18
		temp1 = (((uint64_t)0x6D0125 * (uint64_t)(((uint32_t)src_l1[xh1] + (uint32_t)src_h1[xh1] + (uint32_t)src_l2[xh1] + (uint32_t)src_h2[xh1]) >> 1)) + 0x08000000) >> 28;	// 0x6D01250A // 24 x 18
		temp2 = (((uint64_t)0x6D0125 * (uint64_t)(((uint32_t)src_l1[xl1] + (uint32_t)src_h1[xl1] + (uint32_t)src_l2[xxx] + (uint32_t)src_h2[xxx]) >> 1)) + 0x08000000) >> 28;	// 0x6D01250A // 24 x 18
		temp3 = (((uint64_t)0x59DBE7 * (uint64_t)(((uint32_t)src_l2[xh1] + (uint32_t)src_h2[xh1] + (uint32_t)src_mm[xl1] + (uint32_t)src_mm[xh1]) >> 1)) + 0x04000000) >> 27;	// 0x59DBE71E // 24 x 18
		temp4 = (((uint64_t)0x7A218B * (uint64_t)(((uint32_t)src_l1[xxx] + (uint32_t)src_h1[xxx] + (uint32_t)src_l1[xh1] + (uint32_t)src_h1[xh1]) >> 1)) + 0x02000000) >> 26;	// 0x7A218BA5 // 24 x 18
		temp5 = (((uint64_t)0x64AE15 * (uint64_t)(((uint32_t)src_l1[xh1] + (uint32_t)src_h1[xh1] + (uint32_t)src_mm[xxx] + (uint32_t)src_mm[xh1]) >> 1)) + 0x01000000) >> 25;	// 0x64AE16BF // 24 x 18
		temp6 = (((uint64_t)0x52FF24 * (uint64_t)((uint32_t)src_mm[xh1])) + 0x01000000) >> 25;																					// 0x52FF241B // 24 x 16
		dst_m[xh1] = (uint16_t)(temp0 + temp1 + temp2 + temp3 + temp4 + temp5 + temp6);

	}
}

/**
* @brief This function applies the Scharr operator to the image
*/
static void scharrXY_fp(const uint16_t *Lsmooth, Derivative_fp* Lderiv, const uint8_t scale, const uint16_t scale_N) {

	const uint32_t borderY = img_pixels - (uint32_t)img_cols;	// 19|0
	uint32_t ym;
	for (ym = 0; ym < img_pixels; ym += img_cols) {				// 19|0

		// Y-AXIS BORDER
		int32_t yl = (int32_t)ym - (int32_t)scale_N;			// 20|0
		if (yl < 0) yl = 0;
		uint32_t yh = ym + (uint32_t)scale_N;					// 19|0
		if (yh >= img_pixels) yh = borderY;

		// Y-AXIS POINTER
		const uint16_t* const __restrict Lsmooth_m = &Lsmooth[ym];
		const uint16_t* const __restrict Lsmooth_h = &Lsmooth[yh];
		const uint16_t* const __restrict Lsmooth_l = &Lsmooth[yl];
		Derivative_fp* const __restrict deriv_m = &Lderiv[ym];

		// LEFT BORDER
		uint16_t xx;
		for (xx = 0; xx < scale; xx++) {
			const uint8_t xh = xx + scale;
			const int32_t in0022 = (int32_t)Lsmooth_h[xh] - (int32_t)Lsmooth_l[0];
			const int32_t in02 = (int32_t)Lsmooth_l[xh];
			const int32_t in20 = (int32_t)Lsmooth_h[0];
			const int32_t lx_temp = 3 * (in02 + in0022 - in20) + 10 * ((int32_t)Lsmooth_m[xh] - (int32_t)Lsmooth_m[0]);
			const int32_t ly_temp = 3 * (in20 + in0022 - in02) + 10 * ((int32_t)Lsmooth_h[xx] - (int32_t)Lsmooth_l[xx]);
			deriv_m[xx].Lx = (int16_t)(lx_temp >> 5);
			deriv_m[xx].Ly = (int16_t)(ly_temp >> 5);
		}

		// MIDDLE
		for (xx = scale; xx < img_cols - scale; xx++) {								// 10|0
			const uint16_t xl = xx - (uint16_t)scale;
			const uint16_t xh = xx + (uint16_t)scale;
			const int32_t in0022 = (int32_t)Lsmooth_h[xh] - (int32_t)Lsmooth_l[xl];	// 1|16 - 1|16 = 1|16 (sign extension)
			const int32_t in02 = (int32_t)Lsmooth_l[xh];							// 1|16
			const int32_t in20 = (int32_t)Lsmooth_h[xl];							// 1|16
			// (3 * 2|16) + (10 * 1|16) =  4|16 + 5|16 = 5|16 (signed) Theory(15.9375) max(13.22)
			const int32_t lx_temp = 3 * (in02 + in0022 - in20) + 10 * ((int32_t)Lsmooth_m[xh] - (int32_t)Lsmooth_m[xl]);
			const int32_t ly_temp = 3 * (in20 + in0022 - in02) + 10 * ((int32_t)Lsmooth_h[xx] - (int32_t)Lsmooth_l[xx]);
			// 5|16 >> 5 = 0|16 (signed)
			deriv_m[xx].Lx = (int16_t)(lx_temp >> 5);
			deriv_m[xx].Ly = (int16_t)(ly_temp >> 5);
		}

		// RIGHT Border
		for (xx = img_cols - scale; xx < img_cols; xx++) {
			const uint16_t xl = xx - (uint16_t)scale;
			const uint16_t xh = img_cols - 1;
			const int32_t in0022 = (int32_t)Lsmooth_h[xh] - (int32_t)Lsmooth_l[xl];
			const int32_t in02 = (int32_t)Lsmooth_l[xh];
			const int32_t in20 = (int32_t)Lsmooth_h[xl];
			const int32_t lx_temp = 3 * (in02 + in0022 - in20) + 10 * ((int32_t)Lsmooth_m[xh] - (int32_t)Lsmooth_m[xl]);
			const int32_t ly_temp = 3 * (in20 + in0022 - in02) + 10 * ((int32_t)Lsmooth_h[xx] - (int32_t)Lsmooth_l[xx]);
			deriv_m[xx].Lx = (int16_t)(lx_temp >> 5);
			deriv_m[xx].Ly = (int16_t)(ly_temp >> 5);
		}
	}

/*	for (int xx = 0; xx < img_pixels; xx++) {
		int32_t c = 0;
		c = Lderiv[xx].Ly;
		for (int i = 15; i >= 0; i--) {
			int d = (c >> i) & 1;
			fprintf(LxLy_fp, "%d", d);
		}
		c = Lderiv[xx].Lx;
		for (int i = 15; i >= 0; i--) {
			int d = (c >> i) & 1;
			fprintf(LxLy_fp, "%d", d);
		}
		fprintf(LxLy_fp, "\n");
	}*/
}

/**
* @brief This function computes the determinant of the hessian
*/
static void det_hessian_fp(int32_t *ldet, const Derivative_fp* Lderiv, const uint8_t scale, const uint16_t scale_N) {

	const uint32_t borderY = img_pixels - (uint32_t)img_cols;	// 19|0
	uint32_t ym;
	for (ym = 0; ym < img_pixels; ym += img_cols) {	// 19|0

		// Y-AXIS BORDER
		int32_t yl = (int32_t)ym - (int32_t)scale_N;			// 20|0
		if (yl < 0) yl = 0;
		uint32_t yh = ym + (uint32_t)scale_N;					// 19|0
		if (yh >= img_pixels) yh = borderY;

		// Y-AXIS POINTER
		const Derivative_fp* const __restrict deriv_m = &Lderiv[ym];
		const Derivative_fp* const __restrict deriv_h = &Lderiv[yh];
		const Derivative_fp* const __restrict deriv_l = &Lderiv[yl];
		int32_t* const __restrict Ldet_m = &ldet[ym];

		// LEFT BORDER
		uint16_t xx;
		for (xx = 0; xx < scale; xx++) {
			const uint8_t xh = xx + scale;
			const int32_t lx0022 = (int32_t)(deriv_h[xh].Lx) - (int32_t)(deriv_l[0].Lx);
			const int32_t lx02 = (int32_t)(deriv_l[xh].Lx);
			const int32_t lx20 = (int32_t)(deriv_h[0].Lx);
			const int32_t ly0022 = (int32_t)(deriv_h[xh].Ly) - (int32_t)(deriv_l[0].Ly);
			const int32_t ly02 = (int32_t)(deriv_l[xh].Ly);
			const int32_t ly20 = (int32_t)(deriv_h[0].Ly);
			int32_t lxx = 3 * (lx02 + lx0022 - lx20) + 10 * ((int32_t)(deriv_m[xh].Lx) - (int32_t)(deriv_m[0].Lx));
			int32_t lyy = 3 * (ly20 + ly0022 - ly02) + 10 * ((int32_t)(deriv_h[xx].Ly) - (int32_t)(deriv_l[xx].Ly));
			int32_t lxy = 3 * (lx20 + lx0022 - lx02) + 10 * ((int32_t)(deriv_h[xx].Lx) - (int32_t)(deriv_l[xx].Lx));
			lxx = lxx >> 5;
			lyy = lyy >> 5;
			lxy = lxy >> 5;
			Ldet_m[xx] = (lxx*lyy - lxy*lxy);
		}

		// MIDDLE
		for (xx = scale; xx < img_cols - scale; xx++) {			// 10|0
			const uint16_t xl = xx - (uint16_t)scale;
			const uint16_t xh = xx + (uint16_t)scale;
			const int32_t lx0022 = (int32_t)(deriv_h[xh].Lx) - (int32_t)(deriv_l[xl].Lx);	// 0|16 - 0|16 = 1|16
			const int32_t lx02 = (int32_t)(deriv_l[xh].Lx);									// 0|16
			const int32_t lx20 = (int32_t)(deriv_h[xl].Lx);									// 0|16
			const int32_t ly0022 = (int32_t)(deriv_h[xh].Ly) - (int32_t)(deriv_l[xl].Ly);	// 0|16 - 0|16 = 1|16
			const int32_t ly02 = (int32_t)(deriv_l[xh].Ly);									// 0|16
			const int32_t ly20 = (int32_t)(deriv_h[xl].Ly);									// 0|16
			// (3 * 2|16) + (10 * 1|16) =  4|16 + 5|16 = 5|16 (signed) Theory(under 16) max(8.48)
			int32_t lxx = 3 * (lx02 + lx0022 - lx20) + 10 * ((int32_t)(deriv_m[xh].Lx) - (int32_t)(deriv_m[xl].Lx));
			int32_t lyy = 3 * (ly20 + ly0022 - ly02) + 10 * ((int32_t)(deriv_h[xx].Ly) - (int32_t)(deriv_l[xx].Ly));
			int32_t lxy = 3 * (lx20 + lx0022 - lx02) + 10 * ((int32_t)(deriv_h[xx].Lx) - (int32_t)(deriv_l[xx].Lx));
			// 5|16 >> 5 = 0|16 (signed) max(0.277)
			lxx = lxx >> 5;
			lyy = lyy >> 5;
			lxy = lxy >> 5;
			// 0|16*0|16 - 0|16*0|16 = 0|32 "29" (signed) max(0.036)
			Ldet_m[xx] = (lxx*lyy - lxy*lxy);
		}

		// RIGHT Border
		for (xx = img_cols - scale; xx < img_cols; xx++) {
			const uint16_t xl = xx - (uint16_t)scale;
			const uint16_t xh = img_cols - 1;
			const int32_t lx0022 = (int32_t)(deriv_h[xh].Lx) - (int32_t)(deriv_l[xl].Lx);	// 0|16 - 0|16 = 1|16
			const int32_t lx02 = (int32_t)(deriv_l[xh].Lx);									// 0|16
			const int32_t lx20 = (int32_t)(deriv_h[xl].Lx);									// 0|16
			const int32_t ly0022 = (int32_t)(deriv_h[xh].Ly) - (int32_t)(deriv_l[xl].Ly);	// 0|16 - 0|16 = 1|16
			const int32_t ly02 = (int32_t)(deriv_l[xh].Ly);									// 0|16
			const int32_t ly20 = (int32_t)(deriv_h[xl].Ly);									// 0|16
			// (3 * 2|16) + (10 * 1|16) =  4|16 + 5|16 = 5|16 (signed) Theory(under 16) max(8.48)
			int32_t lxx = 3 * (lx02 + lx0022 - lx20) + 10 * ((int32_t)(deriv_m[xh].Lx) - (int32_t)(deriv_m[xl].Lx));
			int32_t lyy = 3 * (ly20 + ly0022 - ly02) + 10 * ((int32_t)(deriv_h[xx].Ly) - (int32_t)(deriv_l[xx].Ly));
			int32_t lxy = 3 * (lx20 + lx0022 - lx02) + 10 * ((int32_t)(deriv_h[xx].Lx) - (int32_t)(deriv_l[xx].Lx));
			// 5|16 >> 5 = 0|16 (signed) max(0.277)
			lxx = lxx >> 5;
			lyy = lyy >> 5;
			lxy = lxy >> 5;
			// 0|16*0|16 - 0|16*0|16 = 0|32 "29" (signed) max(0.036)
			Ldet_m[xx] = (lxx*lyy - lxy*lxy);
		}
	}

/*	for (int xx = 0; xx < img_pixels; xx++) {
		int32_t c = 0;
		c = ldet[xx];
		for (int i = 31; i >= 0; i--) {
			int d = (c >> i) & 1;
			fprintf(Ldet_fp, "%d", d);
		}
		fprintf(Ldet_fp, "\n");
	}*/
}

/**
* @brief  This function applies the Scharr operator to the image
*         This function computes the Perona and Malik conductivity coefficient
*         g2 = 1 / (1 + dL^2 / k^2)
* @param  src Input image
* @param  Lflow Output image
* @param  contrast_square  square of the contrast factor parameter
*/
static void pm_g2_fp(const uint16_t *Lsmooth, uint16_t* Lflow, const uint32_t contrast_square) {
	
	uint32_t ym;
	for (ym = 0; ym < img_pixels; ym += img_cols) {				// 19|0

		// Y-AXIS BORDER
		int32_t yl = (int32_t)ym - (int32_t)img_cols;			// 20|0
		if (yl < 0) yl = 0;
		uint32_t yh = ym + (uint32_t)img_cols;					// 19|0
		if (yh == img_pixels) yh = ym;

		// Y-AXIS POINTER
		const uint16_t* const __restrict Lsmooth_m = &Lsmooth[ym];
		const uint16_t* const __restrict Lsmooth_h = &Lsmooth[yh];
		const uint16_t* const __restrict Lsmooth_l = &Lsmooth[yl];
		uint16_t* const __restrict Lflow_m = &Lflow[ym];

		// LEFT BORDER
		int32_t in0022 = (int32_t)Lsmooth_h[1] - (int32_t)Lsmooth_l[0];
		int32_t in02 = (int32_t)Lsmooth_l[1];
		int32_t in20 = (int32_t)Lsmooth_h[0];
		int32_t lx = 3 * (in02 + in0022 - in20) + 10 * ((int32_t)Lsmooth_m[1] - (int32_t)Lsmooth_m[0]);
		int32_t ly = 3 * (in20 + in0022 - in02) + 10 * ((int32_t)Lsmooth_h[0] - (int32_t)Lsmooth_l[0]);
		lx = lx >> 5;
		ly = ly >> 5;
		uint32_t contrast_square_ceiling = (contrast_square + 65536) >> 4;
		uint32_t lx2 = ((uint32_t)(lx*lx)) >> 4;
		uint32_t ly2 = ((uint32_t)(ly*ly)) >> 4;
		uint32_t result = (uint32_t)(contrast_square / ((contrast_square_ceiling + lx2 + ly2) >> 12));
		Lflow_m[0] = (uint16_t)result;

		// MIDDLE
		uint16_t xm;
		for (xm = 1; xm < img_cols - 1; xm++) {			// 10|0
			const uint16_t xl = xm - 1;
			const uint16_t xh = xm + 1;
			const int32_t in0022 = (int32_t)Lsmooth_h[xh] - (int32_t)Lsmooth_l[xl];	// 1|16 - 1|16 = 1|16 (sign extension)
			const int32_t in02 = (int32_t)Lsmooth_l[xh];							// 1|16
			const int32_t in20 = (int32_t)Lsmooth_h[xl];							// 1|16
			// (3 * 2|16) + (10 * 1|16) =  4|16 + 5|16 = 5|16 (signed) Theory(15.9375) max(6.96)
			int32_t lx = 3 * (in02 + in0022 - in20) + 10 * ((int32_t)Lsmooth_m[xh] - (int32_t)Lsmooth_m[xl]);
			int32_t ly = 3 * (in20 + in0022 - in02) + 10 * ((int32_t)Lsmooth_h[xm] - (int32_t)Lsmooth_l[xm]);
			// 5|16 >> 5 = 0|16 signed
			lx = lx >> 5;
			ly = ly >> 5;
			// rounding against overflow | 0|32 (unsigned)
			const uint32_t contrast_square_ceiling = (contrast_square + 65536) >> 4;
			const uint32_t lx2 = ((uint32_t)(lx*lx)) >> 4;
			const uint32_t ly2 = ((uint32_t)(ly*ly)) >> 4;
			// 0|16 * 0|16 = 0|32 "28"     (0|32 + 0|32) >> 16 = 0|16     0|32 / 0|16 = 16|16  "1|16"
			const uint32_t result = (uint32_t)(contrast_square / ((contrast_square_ceiling + lx2 + ly2) >> 12));
			// 0|16 (unsigned)
			Lflow_m[xm] = (uint16_t)result;
		}

		// RIGHT Border
		const uint16_t xl = img_cols - 2;
		const uint16_t xx = img_cols - 1;
		in0022 = (int32_t)Lsmooth_h[xx] - (int32_t)Lsmooth_l[xl];
		in02 = (int32_t)Lsmooth_l[xx];
		in20 = (int32_t)Lsmooth_h[xl];
		lx = 3 * (in02 + in0022 - in20) + 10 * ((int32_t)Lsmooth_m[xx] - (int32_t)Lsmooth_m[xl]);
		ly = 3 * (in20 + in0022 - in02) + 10 * ((int32_t)Lsmooth_h[xx] - (int32_t)Lsmooth_l[xx]);
		lx = lx >> 5;
		ly = ly >> 5;
		contrast_square_ceiling = (contrast_square + 65536) >> 4;
		lx2 = ((uint32_t)(lx*lx)) >> 4;
		ly2 = ((uint32_t)(ly*ly)) >> 4;
		result = (uint32_t)(contrast_square / ((contrast_square_ceiling + lx2 + ly2) >> 12));
		Lflow_m[xx] = (uint16_t)result;
	}

/*	for (int xx = 0; xx < img_pixels; xx++) {
		for (int i = 15; i >= 0; i--) {
			fprintf(Lflow_fp, "%d", (Lflow[xx] >> i) % 2);
		}
		fprintf(Lflow_fp, "\n");
	}*/

}

/**
* @brief  This function performs a scalar non-linear diffusion step
* @param  Lt Output image in the evolution
* @param  Lflow Conductivity image
* @param  Lstep Previous image in the evolution
* @param  tsteps The step size in time units
* @note   Forward Euler Scheme 3x3 stencil
* The function c "Lflow" is a scalar value that depends on the gradient norm
* dL_by_ds = d(c dL_by_dx)_by_dx + d(c dL_by_dy)_by_dy
*/
static void nld_step_scalar_fp(uint16_t *Lt, const uint16_t *Lflow, int16_t *Lstep, const uint32_t tsteps) {

	uint32_t ym;
	for (ym = 0; ym < img_pixels; ym += img_cols) {	// 19|0

		// Y-AXIS BORDER
		int32_t yl = (int32_t)ym - (int32_t)img_cols;			// 20|0
		if (yl < 0) yl = 0;
		uint32_t yh = ym + (uint32_t)img_cols;					// 19|0
		if (yh == img_pixels) yh = ym;

		// Y-AXIS POINTER
		const uint16_t* const __restrict Lflow_m = &Lflow[ym];
		const uint16_t* const __restrict Lt_m = &Lt[ym];
		int16_t* const __restrict Lstep_m = &Lstep[ym];

		// LEFT BORDER
		uint16_t Lflow_center = Lflow_m[0];
		uint16_t Lt_center = Lt_m[0];
		int32_t xpos = (((int32_t)Lflow_m[1] + (int32_t)Lflow_center) * ((int32_t)Lt_m[1] - (int32_t)Lt_center) + 2048) >> 12;
		int32_t xneg = (((int32_t)Lflow_m[0] + (int32_t)Lflow_center) * ((int32_t)Lt_m[0] - (int32_t)Lt_center) + 2048) >> 12;
		int32_t ypos = (((int32_t)Lflow[yh] + (int32_t)Lflow_center) * ((int32_t)Lt[yh] - (int32_t)Lt_center) + 2048) >> 12;
		int32_t yneg = (((int32_t)Lflow[yl] + (int32_t)Lflow_center) * ((int32_t)Lt[yl] - (int32_t)Lt_center) + 2048) >> 12;
		int32_t sum = ((xpos + xneg) + (ypos + yneg));
		Lstep_m[0] = (int16_t)(((int64_t)tsteps * (int64_t)sum) >> 19);
		yl++;
		yh++;		

		// MIDDLE
		uint16_t xm;
		for (xm = 1; xm < img_cols - 1; xm++) {			// 10|0
			const uint16_t xl = xm - 1;
			const uint16_t xh = xm + 1;
			uint16_t Lflow_center = Lflow_m[xm];		// 0|16
			uint16_t Lt_center = Lt_m[xm];				// 0|16
			// 2|16 * 1|16 "14" = 0|32 "29"   signed
			int32_t xpos = (((int32_t)Lflow_m[xh] + (int32_t)Lflow_center) * ((int32_t)Lt_m[xh] - (int32_t)Lt_center) + 2048) >> 12;
			int32_t xneg = (((int32_t)Lflow_m[xl] + (int32_t)Lflow_center) * ((int32_t)Lt_m[xl] - (int32_t)Lt_center) + 2048) >> 12;
			int32_t ypos = (((int32_t)Lflow[yh] + (int32_t)Lflow_center) * ((int32_t)Lt[yh] - (int32_t)Lt_center) + 2048) >> 12;
			int32_t yneg = (((int32_t)Lflow[yl] + (int32_t)Lflow_center) * ((int32_t)Lt[yl] - (int32_t)Lt_center) + 2048) >> 12;
			// 1|20 - 1|20 + 1|20 - 1|20 = 0|20
			int32_t sum = ((xpos + xneg) + (ypos + yneg));						// max = 18595, min = -29101
			// 2|20 x 0|18 = (0|38 * 0.5) >> (20+2) = 0|16 "15" (signed)
			Lstep_m[xm] = (int16_t)(((int64_t)tsteps * (int64_t)sum) >> 19);	// 20 + 2 + 1  !!!WARNING evo[15] would overflow "tstep"
			yl++;
			yh++;
		}

		// RIGHT Border
		const uint16_t xl = img_cols - 2;
		const uint16_t xx = img_cols - 1;
		Lflow_center = Lflow_m[xx];
		Lt_center = Lt_m[xx];
		xpos = (((int32_t)Lflow_m[xx] + (int32_t)Lflow_center) * ((int32_t)Lt_m[xx] - (int32_t)Lt_center) + 2048) >> 12;
		xneg = (((int32_t)Lflow_m[xl] + (int32_t)Lflow_center) * ((int32_t)Lt_m[xl] - (int32_t)Lt_center) + 2048) >> 12;
		ypos = (((int32_t)Lflow[yh] + (int32_t)Lflow_center) * ((int32_t)Lt[yh] - (int32_t)Lt_center) + 2048) >> 12;
		yneg = (((int32_t)Lflow[yl] + (int32_t)Lflow_center) * ((int32_t)Lt[yl] - (int32_t)Lt_center) + 2048) >> 12;			
		sum = ((xpos + xneg) + (ypos + yneg));
		Lstep_m[xx] = (int16_t)(((int64_t)tsteps * (int64_t)sum) >> 19);
	}

	// Lt = Lt + Lstep
	uint32_t i;
	for (i = 0; i < img_pixels; i++)					// 19|0		
		Lt[i] = (uint16_t)((int16_t)Lt[i] + (int16_t)Lstep[i]);	// 1|16 + 0|16
}


/**********************************************************************************************************************
******************************************** FUNCTIONS FEATURE DETECTION 1 ********************************************
**********************************************************************************************************************/

/**
* @brief This method computes the initial gaussian and the contrast factor
*/
uint32_t Feature_Detection1_fp(const uint8_t *data) {

	// gaussian smoothing on original image + fixed point conversion (16 bit fraction)   using SIGMA=1 instead SIGMA=1.6 
	gaussian5x5_sigma1_init_fp(data, Scalespace[0].Lt);

	// compute the kcontrast factor
	uint32_t contrast_factor = Compute_KContrast_fp(Scalespace[0].Lt);
	if (contrast_factor >= 4095) contrast_factor = 4095;
	const uint32_t contrast_square = contrast_factor * contrast_factor;

	return contrast_square;
}

/**
* @brief This method computes the k contrast factor
* @param src Input image
* @return k Contrast factor parameter
*/
static uint32_t Compute_KContrast_fp(const uint16_t *Lt) {

	float hmax = 0.0, kperc = 0.0;
	uint32_t nelements = 0, k = 0, npoints = 0;

	// initialized array for Histogram
	uint16_t hist[1024] = { 0 };
	uint16_t bin_max = 0;

	// Skip the borders for computing the histogram and Compute scharr derivatives
	uint32_t ym;
	for (ym = img_cols; ym < (img_pixels - img_cols); ym += img_cols) {

		const uint16_t* const __restrict src_m = &Lt[ym];
		const uint16_t* const __restrict src_h = &Lt[ym + img_cols];
		const uint16_t* const __restrict src_l = &Lt[ym - img_cols];

		uint16_t xm;
		for (xm = 1; xm < img_cols - 1; xm++) {
			const uint16_t xl = xm - 1;
			const uint16_t xh = xm + 1;

			const int32_t in0022 = (int32_t)src_h[xh] - (int32_t)src_l[xl];
			const int32_t in02 = (int32_t)src_l[xh];
			const int32_t in20 = (int32_t)src_h[xl];
			int32_t lx = 3 * (in02 + in0022 - in20) + 10 * ((int32_t)src_m[xh] - (int32_t)src_m[xl]);
			int32_t ly = 3 * (in20 + in0022 - in02) + 10 * ((int32_t)src_h[xm] - (int32_t)src_l[xm]);
			lx = lx >> 5;
			ly = ly >> 5;
			float modg = sqrtf((float)(lx * lx) + (float)(ly * ly));
			// Get the maximum
			if (modg > hmax)
				hmax = modg;

			if (modg != 0.0f) {
				int16_t nbin = (int16_t)floorf(0.03125f*modg); // 2048/(2^16)
				if (nbin > 1023) nbin = 1023;
				hist[nbin]++;
				npoints++;
				if (nbin > bin_max) bin_max = nbin;
			}

		}
	}

	// Now find the perc of the histogram percentile
	const uint32_t nthreshold = (uint32_t)((float)npoints*KCONTRAST_PERCENTILE);
	for (k = 0; nelements < nthreshold && k < bin_max; k++)
		nelements = nelements + hist[k];
	if (nelements < nthreshold) {
		kperc = 0.03f;
	} else {
		kperc = hmax*((float)(k) / (float)bin_max);
	}

	return (uint32_t)(kperc);
}


/**
* @brief  This function smoothes the original an image with a Gaussian kernel
* @param  src Input image
* @param  dst Output image
*/
static void gaussian5x5_sigma1_init_fp(const uint8_t *Img, uint16_t *Lt) {

	const uint32_t borderY = img_pixels - (uint32_t)img_cols;
	uint32_t yym;
	for (yym = 0; yym < img_pixels; yym += img_cols) {

		// Y-AXIS BORDER
		int32_t yl1 = (int32_t)yym - (int32_t)img_cols;
		if (yl1 < 0) yl1 = 0;
		int32_t yl2 = (int32_t)yym - (int32_t)(img_cols << 1);
		if (yl2 < 0) yl2 = 0;
		uint32_t yh1 = yym + (uint32_t)img_cols;
		if (yh1 == img_pixels) yh1 = borderY;
		uint32_t yh2 = yym + (uint32_t)(img_cols << 1);
		if (yh2 >= img_pixels) yh2 = borderY;

		// Y-AXIS POINTER
		const uint8_t* const __restrict src_mm = &Img[yym];
		const uint8_t* const __restrict src_h1 = &Img[yh1];
		const uint8_t* const __restrict src_h2 = &Img[yh2];
		const uint8_t* const __restrict src_l1 = &Img[yl1];
		const uint8_t* const __restrict src_l2 = &Img[yl2];
		uint16_t* const __restrict dst_m = &Lt[yym];

		// LEFT BORDER
		// (<< 8) bit for fixed point conversion in here
		uint16_t temp0 = (0x30A4F5 * ((uint32_t)src_l2[0] + (uint32_t)src_h2[0] + (uint32_t)src_l2[2] + (uint32_t)src_h2[2])) >> 22;
		uint16_t temp1 = (0x1B4049 * ((uint32_t)src_l2[0] + (uint32_t)src_h2[0] + (uint32_t)src_l2[1] + (uint32_t)src_h2[1] + (uint32_t)src_l1[0] + (uint32_t)src_h1[0] + (uint32_t)src_l1[2] + (uint32_t)src_h1[2])) >> 19;
		uint16_t temp2 = (0x2CEDF4 * ((uint32_t)src_l2[0] + (uint32_t)src_h2[0] + (uint32_t)src_mm[0] + (uint32_t)src_mm[2])) >> 19;
		uint16_t temp3 = (0x3D10C6 * ((uint32_t)src_l1[0] + (uint32_t)src_h1[0] + (uint32_t)src_l1[1] + (uint32_t)src_h1[1])) >> 18;
		uint16_t temp4 = (0x32570B * ((uint32_t)src_l1[0] + (uint32_t)src_h1[0] + (uint32_t)src_mm[0] + (uint32_t)src_mm[1])) >> 17;
		uint16_t temp5 = (0xA5FE48 * (uint32_t)(src_mm[0])) >> 18;
		dst_m[0] = temp0 + temp1 + temp2 + temp3 + temp4 + temp5;
		temp0 = (0x30A4F5 * ((uint32_t)src_l2[0] + (uint32_t)src_h2[0] + (uint32_t)src_l2[3] + (uint32_t)src_h2[3])) >> 22;
		temp1 = (0x1B4049 * ((uint32_t)src_l2[0] + (uint32_t)src_h2[0] + (uint32_t)src_l2[2] + (uint32_t)src_h2[2] + (uint32_t)src_l1[0] + (uint32_t)src_h1[0] + (uint32_t)src_l1[3] + (uint32_t)src_h1[3])) >> 19;
		temp2 = (0x2CEDF4 * ((uint32_t)src_l2[1] + (uint32_t)src_h2[1] + (uint32_t)src_mm[0] + (uint32_t)src_mm[3])) >> 19;
		temp3 = (0x3D10C6 * ((uint32_t)src_l1[0] + (uint32_t)src_h1[0] + (uint32_t)src_l1[2] + (uint32_t)src_h1[2])) >> 18;
		temp4 = (0x32570B * ((uint32_t)src_l1[1] + (uint32_t)src_h1[1] + (uint32_t)src_mm[0] + (uint32_t)src_mm[2])) >> 17;
		temp5 = (0xA5FE48 * (uint32_t)(src_mm[1])) >> 18;
		dst_m[1] = temp0 + temp1 + temp2 + temp3 + temp4 + temp5;

		// MIDDLE
		uint16_t xxm;
		for (xxm = 2; xxm < img_cols - 2; xxm++) {
			const uint16_t xl1 = xxm - 1;
			const uint16_t xl2 = xxm - 2;
			const uint16_t xh1 = xxm + 1;
			const uint16_t xh2 = xxm + 2;
			// (<< 8) bit for fixed point conversion in here
			temp0 = (0x30A4F5 * ((uint32_t)src_l2[xl2] + (uint32_t)src_h2[xl2] + (uint32_t)src_l2[xh2] + (uint32_t)src_h2[xh2])) >> 22;
			temp1 = (0x1B4049 * ((uint32_t)src_l2[xl1] + (uint32_t)src_h2[xl1] + (uint32_t)src_l2[xh1] + (uint32_t)src_h2[xh1] + (uint32_t)src_l1[xl2] + (uint32_t)src_h1[xl2] + (uint32_t)src_l1[xh2] + (uint32_t)src_h1[xh2])) >> 19;
			temp2 = (0x2CEDF4 * ((uint32_t)src_l2[xxm] + (uint32_t)src_h2[xxm] + (uint32_t)src_mm[xl2] + (uint32_t)src_mm[xh2])) >> 19;
			temp3 = (0x3D10C6 * ((uint32_t)src_l1[xl1] + (uint32_t)src_h1[xl1] + (uint32_t)src_l1[xh1] + (uint32_t)src_h1[xh1])) >> 18;
			temp4 = (0x32570B * ((uint32_t)src_l1[xxm] + (uint32_t)src_h1[xxm] + (uint32_t)src_mm[xl1] + (uint32_t)src_mm[xh1])) >> 17;
			temp5 = (0xA5FE48 * (uint32_t)(src_mm[xxm])) >> 18;
			dst_m[xxm] = temp0 + temp1 + temp2 + temp3 + temp4 + temp5;
		}

		// RIGHT Border
		const uint16_t xxx = img_cols - 2;
		const uint16_t xl1 = img_cols - 3;
		const uint16_t xl2 = img_cols - 4;
		const uint16_t xh1 = img_cols - 1;
		// (<< 8) bit for fixed point conversion in here
		temp0 = (0x30A4F5 * ((uint32_t)src_l2[xl2] + (uint32_t)src_h2[xl2] + (uint32_t)src_l2[xh1] + (uint32_t)src_h2[xh1])) >> 22;
		temp1 = (0x1B4049 * ((uint32_t)src_l2[xl1] + (uint32_t)src_h2[xl1] + (uint32_t)src_l2[xh1] + (uint32_t)src_h2[xh1] + (uint32_t)src_l1[xl2] + (uint32_t)src_h1[xl2] + (uint32_t)src_l1[xh1] + (uint32_t)src_h1[xh1])) >> 19;
		temp2 = (0x2CEDF4 * ((uint32_t)src_l2[xxx] + (uint32_t)src_h2[xxx] + (uint32_t)src_mm[xl2] + (uint32_t)src_mm[xh1])) >> 19;
		temp3 = (0x3D10C6 * ((uint32_t)src_l1[xl1] + (uint32_t)src_h1[xl1] + (uint32_t)src_l1[xh1] + (uint32_t)src_h1[xh1])) >> 18;
		temp4 = (0x32570B * ((uint32_t)src_l1[xxx] + (uint32_t)src_h1[xxx] + (uint32_t)src_mm[xl1] + (uint32_t)src_mm[xh1])) >> 17;
		temp5 = (0xA5FE48 * (uint32_t)(src_mm[xxx])) >> 18;
		dst_m[xxx] = temp0 + temp1 + temp2 + temp3 + temp4 + temp5;
		temp0 = (0x30A4F5 * ((uint32_t)src_l2[xl1] + (uint32_t)src_h2[xl1] + (uint32_t)src_l2[xh1] + (uint32_t)src_h2[xh1])) >> 22;
		temp1 = (0x1B4049 * ((uint32_t)src_l2[xxx] + (uint32_t)src_h2[xxx] + (uint32_t)src_l2[xh1] + (uint32_t)src_h2[xh1] + (uint32_t)src_l1[xl1] + (uint32_t)src_h1[xl1] + (uint32_t)src_l1[xh1] + (uint32_t)src_h1[xh1])) >> 19;
		temp2 = (0x2CEDF4 * ((uint32_t)src_l2[xh1] + (uint32_t)src_h2[xh1] + (uint32_t)src_mm[xl1] + (uint32_t)src_mm[xh1])) >> 19;
		temp3 = (0x3D10C6 * ((uint32_t)src_l1[xxx] + (uint32_t)src_h1[xxx] + (uint32_t)src_l1[xh1] + (uint32_t)src_h1[xh1])) >> 18;
		temp4 = (0x32570B * ((uint32_t)src_l1[xh1] + (uint32_t)src_h1[xh1] + (uint32_t)src_mm[xxx] + (uint32_t)src_mm[xh1])) >> 17;
		temp5 = (0xA5FE48 * (uint32_t)(src_mm[xh1])) >> 18;
		dst_m[xh1] = temp0 + temp1 + temp2 + temp3 + temp4 + temp5;
	}
}


/**********************************************************************************************************************
********************************************** FUNCTIONS INITIALIZATION **********************************************
*********************************************************************************************************************/

/**
* @brief  This function allocates memory for the nonlinear scale space
*         and computes FED number of cycles and time steps and base data
*/
int16_t kaze_init_fp(void) {

	uint16_t ptr = 0;

#ifndef ZED_BOARD
	// allocating memory for evolution levels
	uint16_t lvl;
	for (lvl = 2; lvl < EVOLUTION_LEVELS; lvl++) {
		if ((Determinant[lvl] = malloc(img_pixels*sizeof(int32_t))) == NULL) return -1;
		if ((Derivatives[lvl] = malloc(img_pixels*sizeof(Derivative_fp))) == NULL) return -1;
	}

	// allocating memory to create nonlinear scale space
	if ((Scalespace[0].Lt = malloc(img_pixels*sizeof(uint16_t))) == NULL) return -1;
	if ((Scalespace[0].Lsmooth = malloc(img_pixels*sizeof(uint16_t))) == NULL) return -1;
	if ((Scalespace[0].Lflow = malloc(img_pixels*sizeof(uint16_t))) == NULL) return -1;
	if ((Scalespace[0].Lstep = malloc(img_pixels*sizeof(int16_t))) == NULL) return -1;
#endif

	// computes the FED number of cycles and time steps and other base data
	float etime[EVOLUTION_LEVELS];
	uint16_t i;
	for (i = 0; i < DEFAULT_OCTAVE_MAX; i++) {
		uint16_t j;
		for (j = 0; j < DEFAULT_NSUBLEVELS; j++) {
			const float esigma = DEFAULT_SCALE_OFFSET*powf((float)2.0, (float)(j) / (float)(DEFAULT_NSUBLEVELS)+i);
			etime[ptr] = 0.5f*(esigma*esigma);
			Evolution[ptr].sigma_size = (uint8_t)(esigma + 0.5f);
			Evolution[ptr].octave = (uint8_t)i;
			Evolution[ptr].sublevel = (uint8_t)j;
			ptr++;
		}
	}
	uint16_t level;
	for (level = 1; level < EVOLUTION_LEVELS; level++) {
		float ttime = etime[level] - etime[level - 1];
		int16_t naux = (int16_t)(ceilf(sqrtf(3.0f*ttime / TAU_MAX + 0.25f) - 0.5f - 1.0e-8f) + 0.5f);
		uint32_t* tau;
		if ((tau = malloc(naux*sizeof(uint32_t))) == NULL) return -1;
		uint16_t steps = (naux <= 0) ? 0 : fed_tau_by_process_time_fp(naux, ttime, tau);
		FED[level].nsteps = (uint8_t)steps;
		FED[level].tsteps = tau;	// 24 bit fraction
	}
	return 0;
}

/**
* @brief This function allocates an array of the least number of time steps such that a certain stopping time
* for the whole process can be obtained and fills it with the respective FED time step sizes for one cycle
* @param n Number of internal steps
* @param T Desired process stopping time
* @param tau The vector with the dynamic step sizes
* @return the number of time steps per cycle or 0 on failure
*/
static int16_t fed_tau_by_process_time_fp(const int16_t n, const float T, uint32_t *tau) {

	float scale = 3.0f*T / (TAU_MAX*(float)(n*(n + 1)));	// Ratio of t we search to maximal t
	float *tauh;				// Helper vector for unsorted taus
	if ((tauh = malloc(n*sizeof(float))) == NULL) return 0;
	float c = 1.0f / (4.0f * (float)n + 2.0f);			// Compute time saver
	float d = scale * TAU_MAX / 2.0f;					// Compute time saver

	// Set up originally ordered tau vector
	int16_t k;
	for (k = 0; k < n; ++k) {
		float h = cosf(PI * (2.0f * (float)k + 1.0f) * c);
		tauh[k] = d / (h * h);
	}

	// Permute list of time steps according to chosen reordering function. This is a heuristic.
	int16_t kappa = n / 2;		// Choose kappa cycle with k = n/2	
	int16_t prime = n + 1;		// Get modulus for permutation

	while (!fed_is_prime_internal_fp(prime))
		prime++;

	// Perform permutation
	int16_t l;
	for (k = 0, l = 0; l < n; k++, l++) {
		int16_t index = 0;
		while ((index = ((k + 1)*kappa) % prime - 1) >= n)
			k++;
		tau[l] = (uint32_t)(tauh[index] * powf(2, 14) + 0.5f);
	}

	free(tauh);
	return n;
}

/**
* @brief This function checks if a number is prime or not
* @param number Number to check if it is prime or not
* @return true if the number is prime
*/
static int16_t fed_is_prime_internal_fp(const int16_t number) {
	if (number <= 1) {
		return 0;
	} else if (number == 1 || number == 2 || number == 3 || number == 5 || number == 7) {
		return 1;
	} else if ((number % 2) == 0 || (number % 3) == 0 || (number % 5) == 0 || (number % 7) == 0) {
		return 0;
	} else {
		int16_t is_prime = 1;
		int16_t upperLimit = (int16_t)sqrtf(number + 1.0f);
		int16_t divisor = 11;
		while (divisor <= upperLimit) {
			if (number % divisor == 0)
				is_prime = 0;
			divisor += 2;
		}
		return is_prime;
	}
}
