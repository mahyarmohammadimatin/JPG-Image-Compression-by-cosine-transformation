## Repository Contents:

### Image Compression Process:
1. Define quantization matrix and cosine transform matrix.
2. Load and preprocess the input image.
3. Extract red, green, and blue channels.
4. Apply cosine transform to each channel.
5. Apply quantization and compress using run-length compression.
6. Calculate compression rate and image quality.
7. Display original and compressed images.

### Image Decompression Process:
1. Decompress run-length compressed images.
2. Apply dequantization and reverse cosine transform.
3. Combine color channels and restore original image.
4. Display decompressed image and compare image quality.

### Results and Analysis:
- Measure compression rate, distance between original and decompressed images.
- Compare image quality using pixel-wise distance and graphical representation.

<img src="https://github.com/mahyarmohammadimatin/JPG-Image-Compression-by-cosine-transformation/blob/main/before-after.PNG" width="300">
