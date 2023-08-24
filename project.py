import numpy as np
from PIL import Image
from math import cos, pi
import matplotlib.pyplot as plt
from itertools import groupby
import sys

def apply_func_chunk(matrix,func,chunk_size=8):
    N, M = matrix.shape
    res_matrix = np.zeros((N, M))
    for i in range(0, N, chunk_size):
        for j in range(0, M, chunk_size):
            chunk = matrix[i:i+chunk_size, j:j+chunk_size]
            res_chunk = func(chunk)
            res_matrix[i:i+chunk_size, j:j+chunk_size] = res_chunk
    return res_matrix

def apply_to_colors(colors,func,chunk=False):
    return [apply_func_chunk(color,func) if chunk else func(color) for color in colors]

def get_C(N): #cosine transform matrix
    return np.array([[np.sqrt(2/N)*cos((pi*i*(2*j+1))/(2*N)) if i else 1/np.sqrt(N) 
      for j in range(N)] for i in range(N)])

def run_length_compression(matrix):
    flat_matrix = matrix.flatten()
    encoded = [(len(list(group)), val) for val, group in groupby(flat_matrix)]
    return encoded
def run_length_decompression(matrix):
    res=[]
    for count,value in matrix:
        res += [value]*count
    return np.array(res).reshape((N,M))

#first, let us define some concepts
quantization_matrix = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                [12, 12, 14, 19, 26, 58, 60, 55],
                                [14, 13, 16, 24, 40, 57, 69, 56],
                                [14, 17, 22, 29, 51, 87, 80, 62],
                                [18, 22, 37, 56, 68, 109, 103, 77],
                                [24, 35, 55, 64, 81, 104, 113, 92],
                                [49, 64, 78, 87, 103, 121, 120, 101],
                                [72, 92, 95, 98, 112, 100, 103, 99]])
CN = get_C(8) 
cosine_transform = lambda matrix: CN.dot(matrix).dot(CN.T)
quantizing = lambda matrix: np.round(matrix/quantization_matrix).astype(int)
dequantizing = lambda matrix: matrix*quantization_matrix
inverse_cosine_transform = lambda matrix: CN.T.dot(matrix).dot(CN)

# Open image
path = 'D:\\'
image = Image.open(path+"IMG_7101.jpg")

# Convert the image to a numpy array multiple of 8 rowwise and columnwise 
image_array = np.array(image)
N,M,_ = image_array.shape
N,M = N-N%8,M-M%8
image_array = image_array[:N,:M]

# step1: Extract the red, green, and blue channels
red_matrix = image_array[:,:,0].astype(int)-127
green_matrix = image_array[:,:,1].astype(int)-127
blue_matrix = image_array[:,:,2].astype(int)-127
colors = [red_matrix,green_matrix,blue_matrix]
        
# step2: Apply cosine transform
colors_dct = apply_to_colors(colors,cosine_transform,chunk=True)

# step3: Apply quantization
colors_quant = apply_to_colors(colors_dct,quantizing,chunk=True)
colors_quant = apply_to_colors(colors_quant,lambda mat:mat.astype(int))

# step4: Comppress using run_length_compression
red = colors_quant[0]
print(f'Zeros rate: {int((np.size(red)-np.count_nonzero(red))*100/np.size(red))}%')
colors_compressed = apply_to_colors(colors_quant,run_length_compression)

# step4: Decomppress
colors_decompressed = apply_to_colors(colors_compressed,run_length_decompression)

# step3-r: Apply dequantization
colors_dequant = apply_to_colors(colors_decompressed,dequantizing,chunk=True)

# step2-r: Apply reverse cosine transform
colors_decomp = apply_to_colors(colors_dequant,inverse_cosine_transform,chunk=True)

# step1-r: Combine colors
colors_decomp_range = apply_to_colors(colors_decomp,lambda mat: (mat+127).astype(np.uint8))
rgb = np.dstack(colors_decomp_range)
compressed_image = Image.fromarray(rgb, "RGB")

# display images and compression rate
default_size = sum([sys.getsizeof(color) for color in colors])
comp_size = sum([sys.getsizeof(color) for color in colors_compressed])
print(f'The compression rate is: {round((default_size-comp_size)*100/default_size)}%')
print(f'Distance of images: {np.sum((colors_decomp[0]-red_matrix)**2)}')
print(f'Distance of each pixel on average: {np.sqrt(np.mean((colors_decomp[0]-red_matrix)**2))}')
print('original image:')
plt.imshow(image)
plt.show()
print('image after decompression:')
plt.imshow(compressed_image)
plt.show()