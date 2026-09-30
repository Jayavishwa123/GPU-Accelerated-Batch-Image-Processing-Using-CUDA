#include <cuda_runtime.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Image {
  int width;
  int height;
  std::vector<unsigned char> pixels;
};

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t error = call;                                              \
    if (error != cudaSuccess) {                                            \
      std::cerr << "CUDA error: " << cudaGetErrorString(error)             \
                << " at " << __FILE__ << ":" << __LINE__ << '\n';          \
      return 1;                                                            \
    }                                                                      \
  } while (0)

__global__ void GaussianBlurKernel(const unsigned char* input,
                                   unsigned char* output, int width,
                                   int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  const int kernel[5][5] = {
      {1, 4, 6, 4, 1},
      {4, 16, 24, 16, 4},
      {6, 24, 36, 24, 6},
      {4, 16, 24, 16, 4},
      {1, 4, 6, 4, 1}};

  int sum = 0;
  int weight_sum = 0;

  for (int ky = -2; ky <= 2; ++ky) {
    for (int kx = -2; kx <= 2; ++kx) {
      int nx = max(0, min(x + kx, width - 1));
      int ny = max(0, min(y + ky, height - 1));
      int weight = kernel[ky + 2][kx + 2];
      sum += input[ny * width + nx] * weight;
      weight_sum += weight;
    }
  }

  output[y * width + x] =
      static_cast<unsigned char>(sum / weight_sum);
}

__global__ void SobelKernel(const unsigned char* input,
                            unsigned char* output, int width,
                            int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
    output[y * width + x] = 0;
    return;
  }

  const int gx[3][3] = {
      {-1, 0, 1},
      {-2, 0, 2},
      {-1, 0, 1}};

  const int gy[3][3] = {
      {-1, -2, -1},
      {0, 0, 0},
      {1, 2, 1}};

  int gradient_x = 0;
  int gradient_y = 0;

  for (int ky = -1; ky <= 1; ++ky) {
    for (int kx = -1; kx <= 1; ++kx) {
      int pixel = input[(y + ky) * width + (x + kx)];
      gradient_x += pixel * gx[ky + 1][kx + 1];
      gradient_y += pixel * gy[ky + 1][kx + 1];
    }
  }

  int magnitude = min(255, abs(gradient_x) + abs(gradient_y));
  output[y * width + x] = static_cast<unsigned char>(magnitude);
}

bool ReadPGM(const std::string& filename, Image* image) {
  std::ifstream file(filename, std::ios::binary);
  if (!file) return false;

  std::string magic;
  file >> magic;
  if (magic != "P5") return false;

  file >> image->width >> image->height;

  int max_value;
  file >> max_value;
  file.get();

  image->pixels.resize(
      static_cast<size_t>(image->width) * image->height);

  file.read(reinterpret_cast<char*>(image->pixels.data()),
            static_cast<std::streamsize>(image->pixels.size()));

  return file.good();
}

bool WritePGM(const std::string& filename, const Image& image) {
  std::ofstream file(filename, std::ios::binary);
  if (!file) return false;

  file << "P5\n";
  file << image.width << " " << image.height << "\n";
  file << "255\n";
  file.write(reinterpret_cast<const char*>(image.pixels.data()),
             static_cast<std::streamsize>(image.pixels.size()));

  return file.good();
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    std::cout << "Usage: ./batch_image_processing "
                 "<input_directory> <output_directory> <count>\n";
    return 0;
  }

  const std::string input_directory = argv[1];
  const std::string output_directory = argv[2];
  const int requested_count = std::stoi(argv[3]);

  fs::create_directories(output_directory);

  std::vector<fs::path> input_files;

  for (const auto& entry : fs::directory_iterator(input_directory)) {
    if (entry.path().extension() == ".pgm") {
      input_files.push_back(entry.path());
    }
  }

  std::sort(input_files.begin(), input_files.end());

  if (input_files.empty()) {
    std::cerr << "No PGM images found in " << input_directory << '\n';
    return 1;
  }

  const int image_count =
      std::min(requested_count, static_cast<int>(input_files.size()));

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));

  if (device_count == 0) {
    std::cerr << "No CUDA GPU detected.\n";
    return 1;
  }

  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));

  std::cout << "CUDA Batch Image Processing\n";
  std::cout << "===========================\n";
  std::cout << "GPU: " << properties.name << '\n';
  std::cout << "Images: " << image_count << '\n';

  Image first_image;
  if (!ReadPGM(input_files[0].string(), &first_image)) {
    std::cerr << "Could not read first input image.\n";
    return 1;
  }

  const int width = first_image.width;
  const int height = first_image.height;
  const size_t image_size = static_cast<size_t>(width) * height;

  unsigned char* d_input = nullptr;
  unsigned char* d_blurred = nullptr;
  unsigned char* d_output = nullptr;

  CUDA_CHECK(cudaMalloc(&d_input, image_size));
  CUDA_CHECK(cudaMalloc(&d_blurred, image_size));
  CUDA_CHECK(cudaMalloc(&d_output, image_size));

  const dim3 block_size(16, 16);
  const dim3 grid_size(
      (width + block_size.x - 1) / block_size.x,
      (height + block_size.y - 1) / block_size.y);

  double total_gpu_time = 0.0;
  int processed = 0;

  for (int i = 0; i < image_count; ++i) {
    Image input_image;
    if (!ReadPGM(input_files[i].string(), &input_image)) {
      std::cerr << "Skipping unreadable image: "
                << input_files[i] << '\n';
      continue;
    }

    if (input_image.width != width || input_image.height != height) {
      std::cerr << "Skipping image with different dimensions: "
                << input_files[i] << '\n';
      continue;
    }

    Image output_image{width, height, std::vector<unsigned char>(image_size)};

    CUDA_CHECK(cudaMemcpy(d_input, input_image.pixels.data(), image_size,
                          cudaMemcpyHostToDevice));

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    GaussianBlurKernel<<<grid_size, block_size>>>(
        d_input, d_blurred, width, height);
    CUDA_CHECK(cudaGetLastError());

    SobelKernel<<<grid_size, block_size>>>(
        d_blurred, d_output, width, height);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    total_gpu_time += milliseconds;

    CUDA_CHECK(cudaMemcpy(output_image.pixels.data(), d_output, image_size,
                          cudaMemcpyDeviceToHost));

    std::ostringstream output_name;
    output_name << output_directory << "/processed_"
                << (i + 1) << ".pgm";

    if (!WritePGM(output_name.str(), output_image)) {
      std::cerr << "Could not write " << output_name.str() << '\n';
    }

    ++processed;

    if (i < 5 || i == image_count - 1) {
      std::cout << "Processed image " << (i + 1)
                << " / " << image_count
                << " | GPU time: " << milliseconds << " ms\n";
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  std::cout << "---------------------------\n";
  std::cout << "Successfully processed: " << processed << '\n';
  std::cout << "Total GPU processing time: "
            << total_gpu_time << " ms\n";

  if (processed > 0) {
    std::cout << "Average GPU time per image: "
              << total_gpu_time / processed << " ms\n";
  }

  std::cout << "Output directory: "
            << output_directory << '\n';

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_blurred));
  CUDA_CHECK(cudaFree(d_output));

  return 0;
}
