This folder contains the source code to run the kNN matching, which matches two images using the KAZE features. The images are namely img1.pgm, img2.pgm, img3.pgm, img4.pgm, img5.pgm and img6.pgm. img1.pgm is used as a train image and rest of the images used as query images. 

Hardware requirements: Personal computer with Intel processors and NVIDIA GPU. 
Software requirements: Open CV 2.4.11, NVIDIA drivers and NVIDIA CUDA Tool Kit. 

The Source code in the folder are: 
1. kaze.c : Program  to extract KAZE features. (used in the project)
2. kaze_fp.c : Program to extract fixed point KAZE feature. 
3. main.cpp : Open CV implementation which uses the Intel Thread Building Blocks to exploit parallelization.
4. main_exp_cpu.cpp : Single thread implementation 
5. main_exp_gpu_single.cu : Implementation using GPU with number_of_keypoints in Train Image threads 
6. main_exp_gpu_double.cu : Implementation using GPU with number_of_keypoints in Train Image multiplied by the number_of_keypoints in Query Image threads 

Commands to compile the program: 

KAZE features: 
Compile the KAZE feature extraction program using "gcc -c kaze.c -o kaze.o"
Compile the KAZE fixed point extraction using "gcc -c kaze_fp.c -o kaze_fp.o"

Compiling main.cpp and main_exp_cpu.cpp :

g++ -I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib/ -g -o main kaze.o kaze_fp.o main.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_stitching

g++ -I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib/ -g -o main_exp_cpu kaze.o kaze_fp.o main_exp_cpu.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_stitching

Compiling main_exp_gpu_single.cu and main_exp_gpu_double.cu 

nvcc -I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib/ -g -o main_exp_gpu_single kaze.o kaze_fp.o main_gpu_single.cu -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_stitching

nvcc -I/usr/local/include/opencv -I/usr/local/include/opencv2 -L/usr/local/lib/ -g -o main_exp_gpu_double kaze.o kaze_fp.o main_exp_gpu_double.cu -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_stitching
