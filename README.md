# GPU-Accelerated-Batch-Image-Processing-Using-CUDA

This project implements GPU-accelerated batch image processing using NVIDIA CUDA. It processes 100 grayscale 256x256 images and applies Gaussian blur followed by Sobel edge detection using custom CUDA kernels.

# Features
* 100-image dataset
* Custom CUDA __global__ kernels
* Gaussian blur on GPU
* Sobel edge detection on GPU
* CUDA event timing
* Command-line arguments
* Generated output images
* Makefile build system
# Requirements
* NVIDIA CUDA-capable GPU
* CUDA Toolkit / nvcc
* Python 3
* C++17 support
# Project Structure
```
cuda-batch-image-processing/
├── src/
│   └── batch_image_processing.cu
├── scripts/
│   └── generate_images.py
├── data/
│   ├── input/
│   └── output/
├── results/
├── Makefile
├── README.md
└── .gitignore
```
# 1. Check CUDA
```
nvcc --version
nvidia-smi
```
# 2. Generate the Dataset
```
python3 scripts/generate_images.py
```
This creates 100 grayscale PGM images at 256x256 resolution.

Check the number of images:
```
ls data/input | wc -l
```
Expected:
```
100
```
# 3. Compile
```
make
```
Or:
```
nvcc -O2 -std=c++17 src/batch_image_processing.cu -o batch_image_processing
```
# 4. Run
```
./batch_image_processing data/input data/output 100
```
Arguments:
```
<input_directory> <output_directory> <number_of_images>
```
Example:
```
./batch_image_processing data/input data/output 100
```
# 5. Save Execution Evidence
```
./batch_image_processing data/input data/output 100 | tee results/execution_log.txt
```
GPU information:
```
nvidia-smi > results/gpu_info.txt
```
Check output count:
```
ls data/output | wc -l
```
Expected:
```
100
```
# CUDA Implementation
The application uses two CUDA kernels:
```
__global__ void GaussianBlurKernel(...)
```
and:
```
__global__ void SobelKernel(...)
```
A 16x16 CUDA thread block is used. Each CUDA thread processes one image pixel.

The processing pipeline is:
```
Input Image
    |
    v
Host to Device Copy
    |
    v
Gaussian Blur CUDA Kernel
    |
    v
Sobel Edge Detection CUDA Kernel
    |
    v
Device to Host Copy
    |
    v
Output PGM Image
```
# Assignment Requirements
This project satisfies the main assignment requirements:

* Public source repository: GitHub
* Large amount of data: 100 images
* Image processing: Gaussian blur and Sobel edge detection
* GPU computation: custom CUDA kernels
* Command-line arguments: supported
Build instructions: Makefile and README
Execution evidence: logs and generated output files
# Important
Run the project in the CUDA laboratory environment before submission. Do not submit fabricated execution results. The exact GPU model and execution time will depend on the available NVIDIA GPU.

# Author
CUDA at Scale Independent Project
