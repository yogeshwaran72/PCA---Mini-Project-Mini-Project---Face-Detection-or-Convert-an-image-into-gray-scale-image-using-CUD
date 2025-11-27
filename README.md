# PCA-Mini-Project---Face-Detection-or-Convert-an-image-into-gray-scale-image-using-CUD
Mini Project - Face Detection or Convert an image into gray scale image using CUDA GPU programming

## REG NO : 212223040249
## NAME : YOGESHWARAN A

## Convert Image to Grayscale image using CUDA

## AIM:

The aim of this project is to demonstrate how to convert an image to grayscale using CUDA programming without relying on the OpenCV library. It serves as an example of GPU-accelerated image processing using CUDA.

## Procedure:

1.Load the input image using the stb_image library.

2.Allocate memory on the GPU for the input and output image buffers.

3.Copy the input image data from the CPU to the GPU.

4.Define a CUDA kernel function that performs the grayscale conversion on each pixel of the image.

5.Launch the CUDA kernel with appropriate grid and block dimensions.

6.Copy the resulting grayscale image data from the GPU back to the CPU.

7.Save the grayscale image using the stb_image_write library.

8.Clean up allocated memory.
## Program:

```
import cv2
from numba import cuda
import sys
from google.colab.patches import cv2_imshow
# Load the image
image = cv2.imread('input.jpeg')
cv2_imshow(image)
# Check if image loading was successful
if image is None:
    print("Error: Unable to load the input image.")
    sys.exit()

# Convert the image to grayscale using CUDA
@cuda.jit
def gpu_rgb_to_gray(input_image, output_image):
    # Calculate the thread's absolute position within the grid
    x, y = cuda.grid(2)
    if x < input_image.shape[0] and y < input_image.shape[1]:
        # Convert RGB to grayscale (simple average)
        gray_value = (input_image[x, y, 0] + input_image[x, y, 1] + input_image[x, y, 2]) / 3
        output_image[x, y] = gray_value

# Allocate GPU memory for the input and output images
d_input = cuda.to_device(image)
d_output = cuda.device_array((image.shape[0], image.shape[1]), dtype=image.dtype)

# Configure the CUDA kernel
threads_per_block = (16, 16)
blocks_per_grid_x = (image.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
blocks_per_grid_y = (image.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

# Launch the CUDA kernel
gpu_rgb_to_gray[blocks_per_grid, threads_per_block](d_input, d_output)

# Copy the grayscale image back to the host memory
grayscale_image = d_output.copy_to_host()

# Display or save the grayscale image
cv2_imshow(grayscale_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

```
## OUTPUT:

Input Image

<img width="1005" height="665" alt="Screenshot 2025-11-01 100533" src="https://github.com/user-attachments/assets/da69cae8-563b-4a1a-b10b-424accf8c038" />


Grayscale Image

<img width="622" height="408" alt="Screenshot 2025-11-01 100550" src="https://github.com/user-attachments/assets/0eb2edf1-1ed2-4035-a6a8-e31edafe2ebc" />


## Result:

The CUDA program successfully converts the input image to grayscale using the GPU. The resulting grayscale image is saved as an output file. This example demonstrates the power of GPU parallelism in accelerating image processing tasks.
