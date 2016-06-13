/***************************************************** INCLUDES ******************************************************/

#include "kaze.h"

/****************************************************** STRUCTS ******************************************************/

typedef struct{
	uint8_t sigma_size;	// Sigma. For linear diffusion t = sigma^2 / 2
	uint8_t octave;		// Image octave
	uint8_t sublevel;	// Image sublevel in each octave
} Evolution_t;

typedef struct {
	uint8_t nsteps;		// Number of steps per cycle
	float* tsteps;		// Vector of FED dynamic time steps
} FED_t;

typedef struct {
	float Lx;			// First order spatial derivative Lx
	float Ly;			// First order spatial derivative Ly
} Derivative_t;

typedef struct {
	float* Lt;			// Evolution image
	float* Lsmooth;		// Smoothed image	
	float* Lflow;		// Diffusivity image
	float* Lstep;		// Evolution step update
} ScaleSpace_t;

/********************************************** GLOBAL VARIABLES/ARRAYS **********************************************/

static Evolution_t Evolution[EVOLUTION_LEVELS];		// Evolution parameters
static FED_t FED[EVOLUTION_LEVELS];					// FED parameters

static Derivative_t* Derivatives[EVOLUTION_LEVELS];	// Lx and Lx Derivatives in scale space
static ScaleSpace_t Scalespace[1];					// Non-linear scale space creation
static float* Determinant[EVOLUTION_LEVELS];		// Detector Responses in scale space

/********************************************** FUNCTIONS Initialization *********************************************/

static int16_t fed_tau_by_process_time(const int16_t n, const float T, float *tau);
static int16_t fed_is_prime_internal(const int16_t number);

/******************************************** FUNCTIONS Feature Detection ********************************************/

static float Compute_KContrast(const float *Lt);
static void gaussian5x5_sigma1_init(const uint8_t *Img, float *Lt);
static void gaussian5x5_sigma1(const float *Lt, float *Lsmooth);
static void scharrXY(const float *Lsmooth, Derivative_t* Lderiv, const uint8_t scale, const uint16_t scale_N);
static void det_hessian(float *Ldet, const Derivative_t* Lderiv, const uint8_t scale, const uint16_t scale_N);
static void pm_g2(const float *Lsmooth, float* Lflow, const float contrast_square);
static void nld_step_scalar(float *Lt, const float *Lflow, float *Lstep, const float tsteps);
static uint32_t Determinant_Hessian_Parallel(const float detector_threshold);
static uint16_t check_maximum_neighbourhood(const float *img, const float value, 
	                                                const uint16_t row, const uint16_t col);
static uint32_t Do_Subpixel_Refinement(const uint32_t number_keypoints);

/******************************************* FUNCTIONS Feature Description *******************************************/

static float Compute_Main_Orientation_SURF(const Derivative_t* derivatives, const uint8_t scale, const float sig_square, 
	                                       const float xf, const float yf);
static void Get_MSURF_Descriptor_64(const Derivative_t* derivatives, const uint8_t scale, const float sig_square, 
	                                const float xf, const float yf, int16_t* desc, float angle);
static float getAngle(const float x, const float y);


/**********************************************************************************************************************
******************************************** FUNCTIONS Feature Description ********************************************
**********************************************************************************************************************/

/**
* @brief  This method computes the set of descriptors through the nonlinear scale space
* @param  number_keypoints Vector of keypoints
* @param  vector for descriptors
*/
void Feature_Description(const uint32_t number_keypoints) {

	uint16_t lvl;
	// Loop to cache efficiently use the derivatives
	for (lvl = 3; lvl < EVOLUTION_LEVELS - 1; lvl++) {

		uint32_t i;
		// Compute one descriptor for each key-point
		for (i = 0; i < number_keypoints; i++) {

			const uint8_t level = key_points[i].level;
			if (level != lvl) continue;			
			const float xf = key_points[i].x_coord;
			const float yf = key_points[i].y_coord;
			const uint8_t scale = key_points[i].scale;
			const float sig_square = -1 / (12.5f*(float)scale*(float)scale);
			const Derivative_t* deriv = Derivatives[level];
			int16_t* desc = &descriptor[i * 64];

			const float angle = Compute_Main_Orientation_SURF(deriv, scale, sig_square, xf, yf);
			Get_MSURF_Descriptor_64(deriv, scale, sig_square, xf, yf, desc, angle);
		}
	}
}

/**
* @brief  This method computes the main orientation for a given keypoint
* @note   The orientation is computed using a similar approach as described in the
*         original SURF method. See Bay et al., Speeded Up Robust Features, ECCV 2006
*/
static float Compute_Main_Orientation_SURF(const Derivative_t* Lderiv, const uint8_t scale, const float sig_square, 
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
				resX[idx] = gweight*Lderiv[ptr].Lx;
				resY[idx] = gweight*Lderiv[ptr].Ly;
				Ang[idx] = getAngle(resX[idx], resY[idx]);
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
	return getAngle(max_sumX, max_sumY);
}

/**
* @brief  This method computes the descriptor of the provided keypoint given the main orientation of the keypoint
* @note   Rectangular grid of 24 s x 24 s. Descriptor Length 64. The descriptor is inspired from Agrawal et al.,
*         CenSurE: Center Surround Extremas for Realtime Feature Detection and Matching, ECCV 2008
*/
static void Get_MSURF_Descriptor_64(const Derivative_t* Lderiv, const uint8_t scale, const float sig_square, 
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

					const float res1x = Lderiv[ptr1].Lx;
					const float res2x = Lderiv[ptr2].Lx;
					const float res3x = Lderiv[ptr3].Lx;
					const float res4x = Lderiv[ptr4].Lx;
					const float res1y = Lderiv[ptr1].Ly;
					const float res2y = Lderiv[ptr2].Ly;
					const float res3y = Lderiv[ptr3].Ly;
					const float res4y = Lderiv[ptr4].Ly;

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
static  float getAngle(const float x, const float y) {
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
* @param  data Input image for which the nonlinear scale space needs to be created
*/
uint32_t Feature_Detection3(const float detector_threshold) {

	// Find scale space extrema
	uint32_t number_keypoints = Determinant_Hessian_Parallel(detector_threshold);

	// Perform some subpixel refinement
	number_keypoints = Do_Subpixel_Refinement(number_keypoints);

	return number_keypoints;
}

/**
* @brief  This method performs the detection of keypoints by using the normalized
*         score of the Hessian determinant through the nonlinear scale space
*/
static uint32_t Determinant_Hessian_Parallel(const float detector_threshold) {

	uint32_t keypoint_cnt1 = 0, keypoint_cnt2 = 0;

	uint16_t lvl;
	for (lvl = 3; lvl < EVOLUTION_LEVELS - 1; lvl++) {
		// skipps borders to check maximum neighbourhood
		uint16_t ix;
		for (ix = 2; ix < img_rows - 2; ix++) {
			const float* const __restrict Ldet_m = &Determinant[lvl][ix*img_cols];
			uint16_t jx;
			for (jx = 2; jx < img_cols - 2; jx++) {
				const float value = Ldet_m[jx];
				// Filter the points with the detector threshold
				// Check the same, lower and upper scale
				if (value > detector_threshold) {
					if (check_maximum_neighbourhood(Determinant[lvl], value, ix, jx) &&
						check_maximum_neighbourhood(Determinant[lvl - 1], value, ix, jx) &&
						check_maximum_neighbourhood(Determinant[lvl + 1], value, ix, jx)) {

						// Add the point of interest!!
						key_points[keypoint_cnt1].x_coord = (float)jx;
						key_points[keypoint_cnt1].y_coord = (float)ix;
						key_points[keypoint_cnt1].value = fabsf(value);
						key_points[keypoint_cnt1].octave = Evolution[lvl].octave;
						key_points[keypoint_cnt1].level = (uint8_t)lvl;
						key_points[keypoint_cnt1].sublevel = Evolution[lvl].sublevel;
						keypoint_cnt1++;

						// Skip the next two, since they can't be maxima
						jx+=2;
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
static uint16_t check_maximum_neighbourhood(const float *img, const float value, 
	                                                const uint16_t row, const uint16_t col) {

	const float* const __restrict img_m = &img[row*img_cols];
	const float* const __restrict img_l = img_m - img_cols;
	const float* const __restrict img_h = img_m + img_cols;
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
static uint32_t Do_Subpixel_Refinement(const uint32_t number_keypoints) {
	int32_t keypoint_cnt = 0;

	uint32_t cnt;
	for (cnt = 0; cnt < number_keypoints; cnt++) {
		const int32_t N = (int32_t)img_cols;
		const int32_t ptr = (int32_t)(key_points[cnt].y_coord)*N + (int32_t)(key_points[cnt].x_coord);

		const float* const __restrict Ldet_m = &Determinant[key_points[cnt].level][ptr];
		const float* const __restrict Ldet_h = &Determinant[key_points[cnt].level + 1][ptr];
		const float* const __restrict Ldet_l = &Determinant[key_points[cnt].level - 1][ptr];

		const float Ldet_lvl_1h = Ldet_m[1];
		const float Ldet_lvl_1l = Ldet_m[-1];
		const float Ldet_lvl_Nh = Ldet_m[N];
		const float Ldet_lvl_Nl = Ldet_m[-N];
		const float Ldet_lvl = *Ldet_m;
		const float Ldet_lvlh = *Ldet_h;
		const float Ldet_lvll = *Ldet_l;

		// Compute the gradient
		const float Dx = 0.5f*(Ldet_lvl_1h - Ldet_lvl_1l);
		const float Dy = 0.5f*(Ldet_lvl_Nh - Ldet_lvl_Nl);
		const float Ds = 0.5f*(Ldet_lvlh - Ldet_lvll);

		// Compute the Hessian
		const float Dxx = Ldet_lvl_1h + Ldet_lvl_1l - 2.0f*Ldet_lvl;
		const float Dyy = Ldet_lvl_Nh + Ldet_lvl_Nl - 2.0f*Ldet_lvl;
		const float Dss = Ldet_lvlh + Ldet_lvll - 2.0f*Ldet_lvl;
		const float Dxy = 0.25f*(Ldet_m[N + 1] + Ldet_m[-N - 1] - Ldet_m[-N + 1] - Ldet_m[N - 1]);
		const float Dxs = 0.25f*(Ldet_h[1] + Ldet_l[-1] - Ldet_h[-1] - Ldet_l[1]);
		const float Dys = 0.25f*(Ldet_h[N] + Ldet_l[-N] - Ldet_h[-N] - Ldet_l[N]);

		// build and solve the linear system with gaussian elimination
		float A[3][4] = { { Dxx, Dxy, Dxs, -Dx }, { Dxy, Dyy, Dys, -Dy }, { Dxs, Dys, Dss, -Ds } };
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
void Festure_Detection2(const float contrast) {

	/*********************************************** TEvolution 1..N-2 ***********************************************/

	uint16_t lvl;
	for (lvl = 2; lvl < EVOLUTION_LEVELS - 1; lvl++) {

		// compute the gaussian smoothing
		gaussian5x5_sigma1(Scalespace[0].Lt, Scalespace[0].Lsmooth);

		// computes the feature detector response for the nonlinear scale space
		// We use the Hessian determinant as feature detector
		const uint8_t scale = Evolution[lvl].sigma_size;
		const uint16_t scale_N = (uint16_t)scale * img_cols;
		scharrXY(Scalespace[0].Lsmooth, Derivatives[lvl], scale, scale_N);
		det_hessian(Determinant[lvl], Derivatives[lvl], scale, scale_N);

		// Compute the conductivity equation and its gaussian derivatives Lx and Ly
		pm_g2(Scalespace[0].Lsmooth, Scalespace[0].Lflow, contrast);

		// Performs n FED inner steps for the next evolution
		uint16_t j;
		for (j = 0; j < FED[lvl].nsteps; j++) {
			nld_step_scalar(Scalespace[0].Lt, Scalespace[0].Lflow, Scalespace[0].Lstep, FED[lvl].tsteps[j]);
		}
	}

	/************************************************ TEvolution N-1 *************************************************/

	// compute the gaussian smoothing
	gaussian5x5_sigma1(Scalespace[0].Lt, Scalespace[0].Lsmooth);

	// computes the feature detector response for the nonlinear scale space
	// We use the Hessian determinant as feature detector
	const uint8_t scale = Evolution[EVOLUTION_LEVELS - 1].sigma_size;
	const uint16_t scale_N = (uint16_t)scale * img_cols;
	scharrXY(Scalespace[0].Lsmooth, Derivatives[EVOLUTION_LEVELS - 1], scale, scale_N);
	det_hessian(Determinant[EVOLUTION_LEVELS - 1], Derivatives[EVOLUTION_LEVELS - 1], scale, scale_N);
}

/**
* @brief This function smoothes an image with a Gaussian kernel
* @param src Input image
* @param dst Output image
*/
static void gaussian5x5_sigma1(const float *Lt, float *Lsmooth) {

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
		const float* const __restrict src_mm = &Lt[yym];
		const float* const __restrict src_h1 = &Lt[yh1];
		const float* const __restrict src_h2 = &Lt[yh2];
		const float* const __restrict src_l1 = &Lt[yl1];
		const float* const __restrict src_l2 = &Lt[yl2];
		float* const __restrict dst_m = &Lsmooth[yym];

		// LEFT BORDER
		float temp0 = 0.0029690167439504977f * (src_l2[0] + src_h2[0] + src_l2[2] + src_h2[2]);
		float temp1 = 0.0133062098910136560f * (src_l2[0] + src_h2[0] + src_l2[1] + src_h2[1] + src_l1[0] + src_h1[0] + src_l1[2] + src_h1[2]);
		float temp2 = 0.0219382312797146460f * (src_l2[0] + src_h2[0] + src_mm[0] + src_mm[2]);
		float temp3 = 0.0596342954361801580f * (src_l1[0] + src_h1[0] + src_l1[1] + src_h1[1]);
		float temp4 = 0.0983203313488457960f * (src_l1[0] + src_h1[0] + src_mm[0] + src_mm[1]);
		float temp5 = 0.1621028216371266900f * (src_mm[0]);
		dst_m[0] = temp0 + temp1 + temp2 + temp3 + temp4 + temp5;
		temp0 = 0.0029690167439504977f * (src_l2[0] + src_h2[0] + src_l2[3] + src_h2[3]);
		temp1 = 0.0133062098910136560f * (src_l2[0] + src_h2[0] + src_l2[2] + src_h2[2] + src_l1[0] + src_h1[0] + src_l1[3] + src_h1[3]);
		temp2 = 0.0219382312797146460f * (src_l2[1] + src_h2[1] + src_mm[0] + src_mm[3]);
		temp3 = 0.0596342954361801580f * (src_l1[0] + src_h1[0] + src_l1[2] + src_h1[2]);
		temp4 = 0.0983203313488457960f * (src_l1[1] + src_h1[1] + src_mm[0] + src_mm[2]);
		temp5 = 0.1621028216371266900f * (src_mm[1]);
		dst_m[1] = temp0 + temp1 + temp2 + temp3 + temp4 + temp5;

		// MIDDLE
		uint16_t xxm;
		for (xxm = 2; xxm < img_cols - 2; xxm++) {
			const uint16_t xl1 = xxm - 1;
			const uint16_t xl2 = xxm - 2;
			const uint16_t xh1 = xxm + 1;
			const uint16_t xh2 = xxm + 2;
			temp0 = 0.0029690167439504977f * (src_l2[xl2] + src_h2[xl2] + src_l2[xh2] + src_h2[xh2]);
			temp1 = 0.0133062098910136560f * (src_l2[xl1] + src_h2[xl1] + src_l2[xh1] + src_h2[xh1] + src_l1[xl2] + src_h1[xl2] + src_l1[xh2] + src_h1[xh2]);
			temp2 = 0.0219382312797146460f * (src_l2[xxm] + src_h2[xxm] + src_mm[xl2] + src_mm[xh2]);
			temp3 = 0.0596342954361801580f * (src_l1[xl1] + src_h1[xl1] + src_l1[xh1] + src_h1[xh1]);
			temp4 = 0.0983203313488457960f * (src_l1[xxm] + src_h1[xxm] + src_mm[xl1] + src_mm[xh1]);
			temp5 = 0.1621028216371266900f * (src_mm[xxm]);
			dst_m[xxm] = temp0 + temp1 + temp2 + temp3 + temp4 + temp5;
		}

		// RIGHT Border
		const uint16_t xh1 = img_cols - 1;
		const uint16_t xxx = img_cols - 2;
		const uint16_t xl1 = img_cols - 3;
		const uint16_t xl2 = img_cols - 4;
		temp0 = 0.0029690167439504977f * (src_l2[xl2] + src_h2[xl2] + src_l2[xh1] + src_h2[xh1]);
		temp1 = 0.0133062098910136560f * (src_l2[xl1] + src_h2[xl1] + src_l2[xh1] + src_h2[xh1] + src_l1[xl2] + src_h1[xl2] + src_l1[xh1] + src_h1[xh1]);
		temp2 = 0.0219382312797146460f * (src_l2[xxx] + src_h2[xxx] + src_mm[xl2] + src_mm[xh1]);
		temp3 = 0.0596342954361801580f * (src_l1[xl1] + src_h1[xl1] + src_l1[xh1] + src_h1[xh1]);
		temp4 = 0.0983203313488457960f * (src_l1[xxx] + src_h1[xxx] + src_mm[xl1] + src_mm[xh1]);
		temp5 = 0.1621028216371266900f * (src_mm[xxx]);
		dst_m[xxx] = temp0 + temp1 + temp2 + temp3 + temp4 + temp5;
		temp0 = 0.0029690167439504977f * (src_l2[xl1] + src_h2[xl1] + src_l2[xh1] + src_h2[xh1]);
		temp1 = 0.0133062098910136560f * (src_l2[xxx] + src_h2[xxx] + src_l2[xh1] + src_h2[xh1] + src_l1[xl1] + src_h1[xl1] + src_l1[xh1] + src_h1[xh1]);
		temp2 = 0.0219382312797146460f * (src_l2[xh1] + src_h2[xh1] + src_mm[xl1] + src_mm[xh1]);
		temp3 = 0.0596342954361801580f * (src_l1[xxx] + src_h1[xxx] + src_l1[xh1] + src_h1[xh1]);
		temp4 = 0.0983203313488457960f * (src_l1[xh1] + src_h1[xh1] + src_mm[xxx] + src_mm[xh1]);
		temp5 = 0.1621028216371266900f * (src_mm[xh1]);
		dst_m[xh1] = temp0 + temp1 + temp2 + temp3 + temp4 + temp5;
	}
}

/**
* @brief This function applies the Scharr operator to the image
*/
static void scharrXY(const float *Lsmooth, Derivative_t* Lderiv, const uint8_t scale, const uint16_t scale_N) {

	const uint32_t borderY = img_pixels - (uint32_t)img_cols;
	uint32_t ym;
	for (ym = 0; ym < img_pixels; ym += img_cols) {

		// Y-AXIS BORDER
		int32_t yl = (int32_t)ym - (int32_t)scale_N;
		if (yl < 0) yl = 0;
		uint32_t yh = ym + (uint32_t)scale_N;
		if (yh >= img_pixels) yh = borderY;

		// Y-AXIS POINTER
		const float* const __restrict src_m = &Lsmooth[ym];
		const float* const __restrict src_h = &Lsmooth[yh];
		const float* const __restrict src_l = &Lsmooth[yl];
		Derivative_t* const __restrict deriv_m = &Lderiv[ym];

		// LEFT BORDER
		uint16_t xx;
		for (xx = 0; xx < scale; xx++) {
			const uint8_t xh = xx + scale;
			const float in0022 = src_h[xh] - src_l[0];
			const float in02 = src_l[xh];
			const float in20 = src_h[0];
			deriv_m[xx].Lx = 0.09375f * (in02 + in0022 - in20) + 0.3125f * (src_m[xh] - src_m[0]);
			deriv_m[xx].Ly = 0.09375f * (in20 + in0022 - in02) + 0.3125f * (src_h[xx] - src_l[xx]);
		}

		// MIDDLE
		for (xx = scale; xx < img_cols - scale; xx++) {
			const uint16_t xl = xx - (uint16_t)scale;
			const uint16_t xh = xx + (uint16_t)scale;
			const float in0022 = src_h[xh] - src_l[xl];
			const float in02 = src_l[xh];
			const float in20 = src_h[xl];
			deriv_m[xx].Lx = 0.09375f * (in02 + in0022 - in20) + 0.3125f * (src_m[xh] - src_m[xl]);
			deriv_m[xx].Ly = 0.09375f * (in20 + in0022 - in02) + 0.3125f * (src_h[xx] - src_l[xx]);
		}

		// RIGHT Border
		for (xx = img_cols - scale; xx < img_cols; xx++) {
			const uint16_t xl = xx - (uint16_t)scale;
			const uint16_t xh = img_cols - 1;
			const float in0022 = src_h[xh] - src_l[xl];
			const float in02 = src_l[xh];
			const float in20 = src_h[xl];
			deriv_m[xx].Lx = 0.09375f * (in02 + in0022 - in20) + 0.3125f * (src_m[xh] - src_m[xl]);
			deriv_m[xx].Ly = 0.09375f * (in20 + in0022 - in02) + 0.3125f * (src_h[xx] - src_l[xx]);
		}
	}
}

/**
* @brief This function computes the determinant of the hessian
*/
static void det_hessian(float *Ldet, const Derivative_t* Lderiv, const uint8_t scale, const uint16_t scale_N) {

	const uint32_t borderY = img_pixels - (uint32_t)img_cols;
	uint32_t ym;
	for (ym = 0; ym < img_pixels; ym += img_cols) {

		// Y-AXIS BORDER
		int32_t yl = (int32_t)ym - (int32_t)scale_N;
		if (yl < 0) yl = 0;
		uint32_t yh = ym + (uint32_t)scale_N;
		if (yh >= img_pixels) yh = borderY;

		// Y-AXIS POINTER
		const Derivative_t* const __restrict deriv_m = &Lderiv[ym];
		const Derivative_t* const __restrict deriv_h = &Lderiv[yh];
		const Derivative_t* const __restrict deriv_l = &Lderiv[yl];
		float* const __restrict Ldet_m = &Ldet[ym];

		// LEFT BORDER
		uint16_t xx;
		for (xx = 0; xx < scale; xx++) {
			const uint8_t xh = xx + scale;
			const float lx0022 = deriv_h[xh].Lx - deriv_l[0].Lx;
			const float lx02 = deriv_l[xh].Lx;
			const float lx20 = deriv_h[0].Lx;
			const float ly0022 = deriv_h[xh].Ly - deriv_l[0].Ly;
			const float ly02 = deriv_l[xh].Ly;
			const float ly20 = deriv_h[0].Ly;
			const float lxx = 0.09375f * (lx02 + lx0022 - lx20) + 0.3125f * (deriv_m[xh].Lx - deriv_m[0].Lx);
			const float lyy = 0.09375f * (ly20 + ly0022 - ly02) + 0.3125f * (deriv_h[xx].Ly - deriv_l[xx].Ly);
			const float lxy = 0.09375f * (lx20 + lx0022 - lx02) + 0.3125f * (deriv_h[xx].Lx - deriv_l[xx].Lx);
			Ldet_m[xx] = (lxx*lyy - lxy*lxy);
		}

		// MIDDLE
		for (xx = scale; xx < img_cols - scale; xx++) {
			const uint16_t xl = xx - (uint16_t)scale;
			const uint16_t xh = xx + (uint16_t)scale;
			const float lx0022 = deriv_h[xh].Lx - deriv_l[xl].Lx;
			const float lx02 = deriv_l[xh].Lx;
			const float lx20 = deriv_h[xl].Lx;
			const float ly0022 = deriv_h[xh].Ly - deriv_l[xl].Ly;
			const float ly02 = deriv_l[xh].Ly;
			const float ly20 = deriv_h[xl].Ly;
			const float lxx = 0.09375f * (lx02 + lx0022 - lx20) + 0.3125f * (deriv_m[xh].Lx - deriv_m[xl].Lx);
			const float lyy = 0.09375f * (ly20 + ly0022 - ly02) + 0.3125f * (deriv_h[xx].Ly - deriv_l[xx].Ly);
			const float lxy = 0.09375f * (lx20 + lx0022 - lx02) + 0.3125f * (deriv_h[xx].Lx - deriv_l[xx].Lx);
			Ldet_m[xx] = (lxx*lyy - lxy*lxy);
		}

		// RIGHT Border
		for (xx = img_cols - scale; xx < img_cols; xx++) {
			const uint16_t xl = xx - (uint16_t)scale;
			const uint16_t xh = img_cols - 1;
			const float lx0022 = deriv_h[xh].Lx - deriv_l[xl].Lx;
			const float lx02 = deriv_l[xh].Lx;
			const float lx20 = deriv_h[xl].Lx;
			const float ly0022 = deriv_h[xh].Ly - deriv_l[xl].Ly;
			const float ly02 = deriv_l[xh].Ly;
			const float ly20 = deriv_h[xl].Ly;
			const float lxx = 0.09375f * (lx02 + lx0022 - lx20) + 0.3125f * (deriv_m[xh].Lx - deriv_m[xl].Lx);
			const float lyy = 0.09375f * (ly20 + ly0022 - ly02) + 0.3125f * (deriv_h[xx].Ly - deriv_l[xx].Ly);
			const float lxy = 0.09375f * (lx20 + lx0022 - lx02) + 0.3125f * (deriv_h[xx].Lx - deriv_l[xx].Lx);
			Ldet_m[xx] = (lxx*lyy - lxy*lxy);
		}
	}
}

/**
* @brief  This function applies the Scharr operator to the image
*         This function computes the Perona and Malik conductivity coefficient 
*         g2 = 1 / (1 + dL^2 / k^2)
* @param  src Input image
* @param  Lflow Output image
* @param  contrast_square  square of the contrast factor parameter
*/
static void pm_g2(const float *Lsmooth, float* Lflow, const float contrast_square) {

	uint32_t ym;
	for (ym = 0; ym < img_pixels; ym += img_cols) {

		// Y-AXIS BORDER
		int32_t yl = (int32_t)ym - (int32_t)img_cols;
		if (yl < 0) yl = 0;
		uint32_t yh = ym + (uint32_t)img_cols;
		if (yh == img_pixels) yh = ym;

		// Y-AXIS POINTER
		const float* const __restrict src_m = &Lsmooth[ym];
		const float* const __restrict src_h = &Lsmooth[yh];
		const float* const __restrict src_l = &Lsmooth[yl];
		float* const __restrict Lflow_m = &Lflow[ym];

		// LEFT BORDER
		float in0022 = src_h[1] - src_l[0];
		float in02   = src_l[1];
		float in20   = src_h[0];
		float lx = 0.09375f * (in02 + in0022 - in20) + 0.3125f * (src_m[1] - src_m[0]);
		float ly = 0.09375f * (in20 + in0022 - in02) + 0.3125f * (src_h[0] - src_l[0]);
		Lflow_m[0] = (contrast_square / (contrast_square + lx * lx + ly * ly));

		// MIDDLE
		uint16_t xm;
		for (xm = 1; xm < img_cols - 1; xm++) {
			const uint16_t xl = xm - 1;
			const uint16_t xh = xm + 1;
			in0022 = src_h[xh] - src_l[xl];
			in02   = src_l[xh];
			in20   = src_h[xl];
			lx = 0.09375f * (in02 + in0022 - in20) + 0.3125f * (src_m[xh] - src_m[xl]);
			ly = 0.09375f * (in20 + in0022 - in02) + 0.3125f * (src_h[xm] - src_l[xm]);
			Lflow_m[xm] = (contrast_square / (contrast_square + lx * lx + ly * ly));
		}

		// RIGHT Border
		const uint16_t xl = img_cols - 2;
		const uint16_t xx = img_cols - 1;
		in0022 = src_h[img_cols - 1] - src_l[xl];
		in02   = src_l[xx];
		in20   = src_h[xl];
		lx = 0.09375f * (in02 + in0022 - in20) + 0.3125f * (src_m[xx] - src_m[xl]);
		ly = 0.09375f * (in20 + in0022 - in02) + 0.3125f * (src_h[xx] - src_l[xx]);
		Lflow_m[xx] = (contrast_square / (contrast_square + lx * lx + ly * ly));
	}
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
static void nld_step_scalar(float *Lt, const float *Lflow, float *Lstep, const float tsteps) {

	uint32_t ym;
	for (ym = 0; ym < img_pixels; ym += img_cols) {

		// Y-AXIS BORDER
		int32_t yl = (int32_t)ym - (int32_t)img_cols;
		if (yl < 0) yl = 0;
		uint32_t yh = ym + (uint32_t)img_cols;
		if (yh == img_pixels) yh = ym;

		// Y-AXIS POINTER
		const float* const __restrict Lflow_m = &Lflow[ym];
		const float* const __restrict Lt_m = &Lt[ym];
		float* const __restrict Lstep_m = &Lstep[ym];

		// LEFT BORDER
		float Lflow_center = Lflow_m[0];
		float Lt_center    = Lt_m[0];
		float xpos = (Lflow_m[1] + Lflow_center) * (Lt_m[1] - Lt_center);
		float xneg = 0;
		float ypos = (Lflow[yh] + Lflow_center) * (Lt[yh] - Lt_center);
		float yneg = (Lflow_center + Lflow[yl]) * (Lt_center - Lt[yl]);
		Lstep_m[0] = 0.5f * tsteps*(xpos - xneg + ypos - yneg);
		yl++;
		yh++;

		// MIDDLE
		uint16_t xm;
		for (xm = 1; xm < img_cols - 1; xm++) {
			const uint16_t xl = xm - 1;
			const uint16_t xh = xm + 1;
			Lflow_center = Lflow_m[xm];
			Lt_center = Lt_m[xm];
			xpos = (Lflow_m[xh] + Lflow_center) * (Lt_m[xh] - Lt_center);
			xneg = (Lflow_center + Lflow_m[xl]) * (Lt_center - Lt_m[xl]);
			ypos = (Lflow[yh] + Lflow_center) * (Lt[yh] - Lt_center);
			yneg = (Lflow_center + Lflow[yl]) * (Lt_center - Lt[yl]);
			Lstep_m[xm] = 0.5f * tsteps*(xpos - xneg + ypos - yneg);
			yl++;
			yh++;
		}

		// RIGHT Border
		const uint16_t xl = img_cols - 2;
		const uint16_t xx = img_cols - 1;
		Lflow_center = Lflow_m[xx];
		Lt_center    = Lt_m[xx];
		xpos = 0;
		xneg = (Lflow_center + Lflow_m[xl]) * (Lt_center - Lt_m[xl]);
		ypos = (Lflow[yh] + Lflow_center) * (Lt[yh] - Lt_center);
		yneg = (Lflow_center + Lflow[yl]) * (Lt_center - Lt[yl]);
		Lstep_m[xx] = 0.5f * tsteps*(xpos - xneg + ypos - yneg);
	}
	
	// Lt = Lt + Lstep 
	uint32_t i;
	for (i = 0; i < img_pixels; i++)
			Lt[i] += Lstep[i];
}


/**********************************************************************************************************************
******************************************** FUNCTIONS FEATURE DETECTION 1 ********************************************
**********************************************************************************************************************/

/**
* @brief This method computes the initial gaussian and the contrast factor
*/
float Feature_Detection1(const uint8_t *data) {

	// gaussian smoothing on original image + image conversion | using SIGMA=1 instead SIGMA=1.6
	gaussian5x5_sigma1_init(data, Scalespace[0].Lt);

	// compute the kcontrast factor
	const float contrast_factor = Compute_KContrast(Scalespace[0].Lt);
	const float contrast_square = contrast_factor*contrast_factor;

	return contrast_square;
}

/**
* @brief This method computes the k contrast factor
* @param src Input image
* @return k Contrast factor parameter
*/
static float Compute_KContrast(const float *Lt) {

	float hmax = 0.0, kperc = 0.0;
	uint32_t nelements = 0, k = 0, npoints = 0;

	// initialized array for Histogram
	uint16_t hist[1024] = { 0 };
	uint16_t bin_max = 0;

	// Skip the borders for computing the histogram and Compute scharr derivatives
	uint32_t ym;
	for (ym = img_cols; ym < (img_pixels - img_cols); ym += img_cols) {

		const float* const __restrict src_m = &Lt[ym];
		const float* const __restrict src_h = &Lt[ym + img_cols];
		const float* const __restrict src_l = &Lt[ym - img_cols];

		uint16_t xm;
		for (xm = 1; xm < img_cols - 1; xm++) {
			const uint16_t xl = xm - 1;
			const uint16_t xh = xm + 1;

			float in0022 = src_h[xh] - src_l[xl];
			float in02 = src_l[xh];
			float in20 = src_h[xl];
			float Lx = 0.09375f * (in02 + in0022 - in20) + 0.3125f * (src_m[xh] - src_m[xl]);
			float Ly = 0.09375f * (in20 + in0022 - in02) + 0.3125f * (src_h[xm] - src_l[xm]);
			float modg = sqrtf(Lx*Lx + Ly*Ly);
			// Get the maximum
			if (modg > hmax)
				hmax = modg;
			if (modg != 0.0) {
				int16_t nbin = (int16_t)floorf(2048 * modg); // 1024 / 2
				if (nbin > 1023) nbin = 1023;
				hist[nbin]++;
				npoints++;
				if (nbin > bin_max)
					bin_max = nbin;
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

	return kperc;
}

/**
* @brief  This function smoothes the original an image with a Gaussian kernel
* @param  src Input image
* @param  dst Output image
*/
static void gaussian5x5_sigma1_init(const uint8_t *Img, float *Lt) {

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
		float* const __restrict dst_m = &Lt[yym];

		// LEFT BORDER
		float temp0 = 0.000011597721656056631f * (float)((uint16_t)src_l2[0] + (uint16_t)src_h2[0] + (uint16_t)src_l2[2] + (uint16_t)src_h2[2]);
		float temp1 = 0.000051977382386772095f * (float)((uint16_t)src_l2[0] + (uint16_t)src_h2[0] + (uint16_t)src_l2[1] + (uint16_t)src_h2[1] + (uint16_t)src_l1[0] + (uint16_t)src_h1[0] + (uint16_t)src_l1[2] + (uint16_t)src_h1[2]);
		float temp2 = 0.000085696215936385337f * (float)((uint16_t)src_l2[0] + (uint16_t)src_h2[0] + (uint16_t)src_mm[0] + (uint16_t)src_mm[2]);
		float temp3 = 0.000232946466547578740f * (float)((uint16_t)src_l1[0] + (uint16_t)src_h1[0] + (uint16_t)src_l1[1] + (uint16_t)src_h1[1]);
		float temp4 = 0.000384063794331428890f * (float)((uint16_t)src_l1[0] + (uint16_t)src_h1[0] + (uint16_t)src_mm[0] + (uint16_t)src_mm[1]);
		float temp5 = 0.000633214147020026140f * (float)(src_mm[0]);
		dst_m[0] = temp0 + temp1 + temp2 + temp3 + temp4 + temp5;
		temp0 = 0.000011597721656056631f * (float)((uint16_t)src_l2[0] + (uint16_t)src_h2[0] + (uint16_t)src_l2[3] + (uint16_t)src_h2[3]);
		temp1 = 0.000051977382386772095f * (float)((uint16_t)src_l2[0] + (uint16_t)src_h2[0] + (uint16_t)src_l2[2] + (uint16_t)src_h2[2] + (uint16_t)src_l1[0] + (uint16_t)src_h1[0] + (uint16_t)src_l1[3] + (uint16_t)src_h1[3]);
		temp2 = 0.000085696215936385337f * (float)((uint16_t)src_l2[1] + (uint16_t)src_h2[1] + (uint16_t)src_mm[0] + (uint16_t)src_mm[3]);
		temp3 = 0.000232946466547578740f * (float)((uint16_t)src_l1[0] + (uint16_t)src_h1[0] + (uint16_t)src_l1[2] + (uint16_t)src_h1[2]);
		temp4 = 0.000384063794331428890f * (float)((uint16_t)src_l1[1] + (uint16_t)src_h1[1] + (uint16_t)src_mm[0] + (uint16_t)src_mm[2]);
		temp5 = 0.000633214147020026140f * (float)(src_mm[1]);
		dst_m[1] = temp0 + temp1 + temp2 + temp3 + temp4 + temp5;

		// MIDDLE
		uint16_t xxm;
		for (xxm = 2; xxm < img_cols - 2; xxm++) {
			const uint16_t xl1 = xxm - 1;
			const uint16_t xl2 = xxm - 2;
			const uint16_t xh1 = xxm + 1;
			const uint16_t xh2 = xxm + 2;
			temp0 = 0.000011597721656056631f * (float)((uint16_t)src_l2[xl2] + (uint16_t)src_h2[xl2] + (uint16_t)src_l2[xh2] + (uint16_t)src_h2[xh2]);
			temp1 = 0.000051977382386772095f * (float)((uint16_t)src_l2[xl1] + (uint16_t)src_h2[xl1] + (uint16_t)src_l2[xh1] + (uint16_t)src_h2[xh1] + (uint16_t)src_l1[xl2] + (uint16_t)src_h1[xl2] + (uint16_t)src_l1[xh2] + (uint16_t)src_h1[xh2]);
			temp2 = 0.000085696215936385337f * (float)((uint16_t)src_l2[xxm] + (uint16_t)src_h2[xxm] + (uint16_t)src_mm[xl2] + (uint16_t)src_mm[xh2]);
			temp3 = 0.000232946466547578740f * (float)((uint16_t)src_l1[xl1] + (uint16_t)src_h1[xl1] + (uint16_t)src_l1[xh1] + (uint16_t)src_h1[xh1]);
			temp4 = 0.000384063794331428890f * (float)((uint16_t)src_l1[xxm] + (uint16_t)src_h1[xxm] + (uint16_t)src_mm[xl1] + (uint16_t)src_mm[xh1]);
			temp5 = 0.000633214147020026140f * (float)(src_mm[xxm]);
			dst_m[xxm] = temp0 + temp1 + temp2 + temp3 + temp4 + temp5;
		}

		// RIGHT Border
		const uint16_t xh1 = img_cols - 1;
		const uint16_t xxx = img_cols - 2;
		const uint16_t xl1 = img_cols - 3;
		const uint16_t xl2 = img_cols - 4;
		temp0 = 0.000011597721656056631f * (float)((uint16_t)src_l2[xl2] + (uint16_t)src_h2[xl2] + (uint16_t)src_l2[xh1] + (uint16_t)src_h2[xh1]);
		temp1 = 0.000051977382386772095f * (float)((uint16_t)src_l2[xl1] + (uint16_t)src_h2[xl1] + (uint16_t)src_l2[xh1] + (uint16_t)src_h2[xh1] + (uint16_t)src_l1[xl2] + (uint16_t)src_h1[xl2] + (uint16_t)src_l1[xh1] + (uint16_t)src_h1[xh1]);
		temp2 = 0.000085696215936385337f * (float)((uint16_t)src_l2[xxx] + (uint16_t)src_h2[xxx] + (uint16_t)src_mm[xl2] + (uint16_t)src_mm[xh1]);
		temp3 = 0.000232946466547578740f * (float)((uint16_t)src_l1[xl1] + (uint16_t)src_h1[xl1] + (uint16_t)src_l1[xh1] + (uint16_t)src_h1[xh1]);
		temp4 = 0.000384063794331428890f * (float)((uint16_t)src_l1[xxx] + (uint16_t)src_h1[xxx] + (uint16_t)src_mm[xl1] + (uint16_t)src_mm[xh1]);
		temp5 = 0.000633214147020026140f * (float)(src_mm[xxx]);
		dst_m[xxx] = temp0 + temp1 + temp2 + temp3 + temp4 + temp5;
		temp0 = 0.000011597721656056631f * (float)((uint16_t)src_l2[xl1] + (uint16_t)src_h2[xl1] + (uint16_t)src_l2[xh1] + (uint16_t)src_h2[xh1]);
		temp1 = 0.000051977382386772095f * (float)((uint16_t)src_l2[xxx] + (uint16_t)src_h2[xxx] + (uint16_t)src_l2[xh1] + (uint16_t)src_h2[xh1] + (uint16_t)src_l1[xl1] + (uint16_t)src_h1[xl1] + (uint16_t)src_l1[xh1] + (uint16_t)src_h1[xh1]);
		temp2 = 0.000085696215936385337f * (float)((uint16_t)src_l2[xh1] + (uint16_t)src_h2[xh1] + (uint16_t)src_mm[xl1] + (uint16_t)src_mm[xh1]);
		temp3 = 0.000232946466547578740f * (float)((uint16_t)src_l1[xxx] + (uint16_t)src_h1[xxx] + (uint16_t)src_l1[xh1] + (uint16_t)src_h1[xh1]);
		temp4 = 0.000384063794331428890f * (float)((uint16_t)src_l1[xh1] + (uint16_t)src_h1[xh1] + (uint16_t)src_mm[xxx] + (uint16_t)src_mm[xh1]);
		temp5 = 0.000633214147020026140f * (float)(src_mm[xh1]);
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
int16_t kaze_init(void) {
	
	uint16_t ptr = 0;

	// allocating memory for evolution levels
	uint16_t lvl;
	for (lvl = 2; lvl < EVOLUTION_LEVELS; lvl++) {
		if ((Derivatives[lvl] = (Derivative_t *)malloc(img_pixels*sizeof(Derivative_t))) == NULL) return -1;
		if ((Determinant[lvl] = (float *)malloc(img_pixels*sizeof(float))) == NULL) return -1;
	}

	// allocating memory to create nonlinear scale space
	if ((Scalespace[0].Lt = (float *)malloc(img_pixels*sizeof(float))) == NULL) return -1;
	if ((Scalespace[0].Lflow = (float *)malloc(img_pixels*sizeof(float))) == NULL) return -1;
	if ((Scalespace[0].Lstep = (float *)malloc(img_pixels*sizeof(float))) == NULL) return -1;
	if ((Scalespace[0].Lsmooth = (float *)malloc(img_pixels*sizeof(float))) == NULL) return -1;

	// Evolution time
	float etime[EVOLUTION_LEVELS];

	// computes the FED number of cycles and time steps and other base data
	uint16_t i;
	for (i = 0; i <= DEFAULT_OCTAVE_MAX - 1; i++) {
		uint16_t j;
		for (j = 0; j <= DEFAULT_NSUBLEVELS - 1; j++) {
			const float esigma = DEFAULT_SCALE_OFFSET*powf(2.0f, (float)(j) / (float)(DEFAULT_NSUBLEVELS)+i);
			etime[ptr] = 0.5f*(esigma*esigma);
			Evolution[ptr].sigma_size = (uint8_t)(esigma + 0.5);
			Evolution[ptr].octave = (uint8_t)i;
			Evolution[ptr].sublevel = (uint8_t)j;
			ptr++;
		}
	}

	uint16_t level;
	for (level = 1; level < EVOLUTION_LEVELS; level++) {
		float ttime = etime[level] - etime[level - 1];
		int16_t naux = (int16_t)(ceilf(sqrtf(3.0f*ttime / TAU_MAX + 0.25f) - 0.5f - 1.0e-8f) + 0.5f);
		float* tau;
		if ((tau = (float *)malloc(naux*sizeof(float))) == NULL) return -1;
		uint16_t steps = (naux <= 0) ? 0 : fed_tau_by_process_time(naux, ttime, tau);
		FED[level].nsteps = (uint8_t)steps;
		FED[level].tsteps = tau;
	}
	return 0;
}

void kaze_exit(void) {

	uint16_t lvl;
	for (lvl = 2; lvl < EVOLUTION_LEVELS; lvl++) {
		free(Derivatives[lvl]);
		free(Determinant[lvl]);
	}
	free(Scalespace[0].Lt);
	free(Scalespace[0].Lflow);
	free(Scalespace[0].Lstep);
	free(Scalespace[0].Lsmooth);
}

/**
* @brief This function allocates an array of the least number of time steps such that a certain stopping time
* for the whole process can be obtained and fills it with the respective FED time step sizes for one cycle
* @param n Number of internal steps
* @param T Desired process stopping time
* @param tau The vector with the dynamic step sizes
* @return the number of time steps per cycle or 0 on failure
*/
static int16_t fed_tau_by_process_time(const int16_t n, const float T, float *tau) {

	float scale = 3.0f*T / (TAU_MAX*(float)(n*(n + 1)));	// Ratio of t we search to maximal t
	float *tauh;											// Helper vector for unsorted taus
	if ((tauh = (float *)malloc(n*sizeof(float))) == NULL) return -1;
	float c = 1.0f / (4.0f * (float)n + 2.0f);				// Compute time saver
	float d = scale * TAU_MAX / 2.0f;						// Compute time saver

	// Set up originally ordered tau vector
	int16_t k;
	for (k = 0; k < n; k++) {
		float h = cosf(PI * (2.0f * (float)k + 1.0f) * c);
		tauh[k] = d / (h * h);
	}

	// Permute list of time steps according to chosen reordering function. This is a heuristic.
	int16_t kappa = n / 2;		// Choose kappa cycle with k = n/2	
	int16_t prime = n + 1;		// Get modulus for permutation

	while (!fed_is_prime_internal(prime))
		prime++;

	// Perform permutation
	int16_t l;
	for (k = 0, l = 0; l < n; k++, l++) {
		int16_t index = 0;
		while ((index = ((k + 1)*kappa) % prime - 1) >= n) {
			k++;
		}			
		tau[l] = tauh[index];
	}

	free(tauh);
	return n;
}

/**
* @brief This function checks if a number is prime or not
* @param number Number to check if it is prime or not
* @return true if the number is prime
*/
static int16_t fed_is_prime_internal(const int16_t number) {
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
