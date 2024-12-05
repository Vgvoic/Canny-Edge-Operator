#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include <string.h>

#define RGB_VALUE 255
#define kernel_size 2
#define sigma_constant 2
#define double_high_treshold 0.6
#define double_low_treshold 0.2

typedef struct {
	unsigned char red;
	unsigned char green;
	unsigned char blue;
} PPMPixel;

typedef struct {
	int x;
	int y;
	PPMPixel *data;
} PPMImg;

static PPMImg *read_image(const char *filename);
int **create_gaussian_kernel_filter(int sigma, int size);
PPMImg *create_grayscale(PPMImg *colorImage);
PPMImg *sobel_gradient(PPMImg *image);
PPMImg *non_maximum_suppression(PPMImg *gradientMagnitude, PPMImg *gradientOrientation);
PPMImg *double_threshold(PPMImg *gradientMagnitude, float highThresholdPercent, float lowThresholdPercent);
PPMImg *hysteresis(PPMImg *image);
void write_image_to_file(const char *filename, PPMImg *img);


PPMImg *gradient_orientation(PPMImg *img) {

    PPMImg *result;

    result = (PPMImg *)malloc(sizeof(PPMImg));
    result->x = img->x;
    result->y = img->y;
    result->data = (PPMPixel *)malloc(img->x * img->y * sizeof(PPMPixel));

    for(int i = 1; i < img->y - 1; i++) {
        for(int j = 1; j < img->x - 1; j++) {
            int idx = i * img->x + j;
            int idx_up = (i - 1) * img->x + j;
            int idx_down = (i + 1) * img->x + j;
            int idx_left = i * img->x + (j - 1);
            int idx_right = i * img->x + (j + 1);

            float dx = (img->data[idx_right].red + img->data[idx_right].green + img->data[idx_right].blue) -
                        (img->data[idx_left].red + img->data[idx_left].green + img->data[idx_left].blue);

            float dy = (img->data[idx_down].red + img->data[idx_down].green + img->data[idx_down].blue) -
                        (img->data[idx_up].red + img->data[idx_up].green + img->data[idx_up].blue);

            float tan_res = atan2(dy, dx) * 180 / M_PI;

            if (tan_res <= 22.5 || tan_res > 157.5) tan_res = 0.0;
            else if (tan_res > 22.5 && tan_res <= 67.5) tan_res = 45.0;
            else if (tan_res > 67.5 && tan_res <= 112.5) tan_res = 90.0;
            else if (tan_res > 112.5 && tan_res <= 157.5) tan_res = 135.0;

            result->data[idx].red = result->data[idx].green = result->data[idx].blue = tan_res;
        }
    }
    return result;
}


PPMImg *double_threshold(PPMImg *gradientMagnitude, float highThresholdPercent, float lowThresholdPercent) {
    PPMImg *result = (PPMImg *)malloc(sizeof(PPMImg));
    result->x = gradientMagnitude->x;
    result->y = gradientMagnitude->y;
    result->data = (PPMPixel *)malloc(result->x * result->y * sizeof(PPMPixel));

    float maxMagnitude = 0.0;
    for (int i = 0; i < gradientMagnitude->x * gradientMagnitude->y; i++) {
        float magnitude = gradientMagnitude->data[i].red;
        if (magnitude > maxMagnitude) {
            maxMagnitude = magnitude;
        }
    }

    float highThreshold = highThresholdPercent * maxMagnitude;
    float lowThreshold = lowThresholdPercent * maxMagnitude;

    for (int i = 0; i < result->x; i++) {
        for (int j = 0; j < result->y; j++) {
            float magnitude = gradientMagnitude->data[i * result->y + j].red;
            if (magnitude > highThreshold) {
                result->data[i * result->y + j].red = 255;
                result->data[i * result->y + j].green = 255;
                result->data[i * result->y + j].blue = 255;
            } else if (magnitude <= highThreshold && magnitude > lowThreshold) {
                result->data[i * result->y + j].red = 128;
                result->data[i * result->y + j].green = 128;
                result->data[i * result->y + j].blue = 128;
            } else {
                result->data[i * result->y + j].red = 0;
                result->data[i * result->y + j].green = 0;
                result->data[i * result->y + j].blue = 0;
            }
        }
    }

    return result;
}


PPMImg *hysteresis(PPMImg *image) {
    PPMImg *result = (PPMImg *)malloc(sizeof(PPMImg));
    result->x = image->x;
    result->y = image->y;
    result->data = (PPMPixel *)malloc(image->x * image->y * sizeof(PPMPixel));
    float maxVal = 0.f;


    for (int i = 0; i < image->y; i++) {
        for (int j = 0; j < image->x; j++) {
            if (image->data[i * image->x + j].red > maxVal || image->data[i * image->x + j].green > maxVal || image->data[i * image->x + j].blue > maxVal) {
                maxVal = image->data[i * image->x + j].red;
            }
        }
    }

    for (int i = 0; i < image->y; i++) {
        for (int j = 0; j < image->x; j++) {
            if (image->data[i * image->x + j].red < maxVal || image->data[i * image->x + j].green < maxVal || image->data[i * image->x + j].blue < maxVal) {
                result->data[i * result->x + j].red = 0;
                result->data[i * result->x + j].green = 0;
                result->data[i * result->x + j].blue = 0;
            } else {
                result->data[i * result->x + j].red = image->data[i * result->x + j].red;
                result->data[i * result->x + j].green = image->data[i * result->x + j].green;
                result->data[i * result->x + j].blue = image->data[i * result->x + j].blue;
            }
        }
    }

    return result;
}


PPMImg *non_maximum_suppression(PPMImg *grad_mag, PPMImg *grad_orient) {
    PPMImg *result;

    result = (PPMImg *)malloc(sizeof(PPMImg));
    result->x = grad_mag->x;
    result->y = grad_mag->y;
    result->data = (PPMPixel *)malloc(grad_mag->x * grad_mag->y * sizeof(PPMPixel));

    for (int i = 0; i < grad_mag->y; i++) {
        for (int j = 0; j < grad_mag->x; j++) {
            result->data[i * grad_mag->x + j].red = 0;
            result->data[i * grad_mag->x + j].green = 0;
            result->data[i * grad_mag->x + j].blue = 0;
        }
    }

    for(int i = 1; i < grad_mag->y - 1; i++) {
        for(int j = 1; j < grad_mag->x - 1; j++) {
            int idx = i * grad_mag->x + j;

            float angle = grad_orient->data[idx].red;
            float magnitude = grad_mag->data[idx].red;

            float magnitude1, magnitude2;
            if (angle == 45.0) {
                magnitude1 = grad_mag->data[idx - grad_mag->x - 1].red;
                magnitude2 = grad_mag->data[idx + grad_mag->x + 1].red;
            } else if (angle == 90.0) {
                magnitude1 = grad_mag->data[idx - grad_mag->x].red;
                magnitude2 = grad_mag->data[idx + grad_mag->x].red;
            } else if (angle == 135.0) {
                magnitude1 = grad_mag->data[idx - grad_mag->x + 1].red;
                magnitude2 = grad_mag->data[idx + grad_mag->x - 1].red;
            } else {
                magnitude1 = grad_mag->data[idx - 1].red;
                magnitude2 = grad_mag->data[idx + 1].red;
            }

            if (magnitude >= magnitude1 && magnitude >= magnitude2) {
                result->data[idx].red = result->data[idx].green = result->data[idx].blue = magnitude;
            } else {
                result->data[idx].red = result->data[idx].green = result->data[idx].blue = 0;
            }
        }
    }

    return result;
}


int compare(const void *a, const void *b) {
    float fa = *(const float*) a;
    float fb = *(const float*) b;
    return (fa > fb) - (fa < fb);
}


PPMImg *sobel_gradient(PPMImg *grayscaleImg) {

    float kernelx[3][3] = {{1, 0, -1},
                           {2, 0, -2},
                           {1, 0, -1}};

    float kernely[3][3] = {{1, 2, 1},
                           {0,  0,  0},
                           {-1,  -2, -1}};

    PPMImg *result = (PPMImg *)malloc(sizeof(PPMImg));
    result->x = grayscaleImg->x;
    result->y = grayscaleImg->y;
    result->data = (PPMPixel *)malloc(result->x * result->y * sizeof(PPMPixel));

    float *gradientMagnitudes = (float *)malloc(result->x * result->y * sizeof(float));

    int half_kernel_size = 3 / 2;

    for (int i = 0; i < grayscaleImg->y; i++) {
        for (int j = 0; j < grayscaleImg->x; j++) {
            float sumX = 0;
            float sumY = 0;

            for (int m = -half_kernel_size; m <= half_kernel_size; m++) {
                for (int n = -half_kernel_size; n <= half_kernel_size; n++) {

                    if (i + m >= 0 && i + m < grayscaleImg->y && j + n >= 0 && j + n < grayscaleImg->x) {
                        int intensity = grayscaleImg->data[(i + m) * grayscaleImg->x + (j + n)].red;

                        sumX += intensity * kernely[m + half_kernel_size][n + half_kernel_size];
                        sumY += intensity * kernelx[m + half_kernel_size][n + half_kernel_size];
                    }
                }
            }

            float gradientMagnitude = sqrt(sumX * sumX + sumY * sumY);
            gradientMagnitudes[i * result->x + j] = gradientMagnitude;
        }
    }

    float percentile = 0.94;

    float *sortedGradientMagnitudes = (float *)malloc(result->x * result->y * sizeof(float));
    memcpy(sortedGradientMagnitudes, gradientMagnitudes, result->x * result->y * sizeof(float));
    qsort(sortedGradientMagnitudes, result->x * result->y, sizeof(float), compare);
    float threshold = sortedGradientMagnitudes[(int)(percentile * result->x * result->y)];

    for (int i = 0; i < result->y; i++) {
        for (int j = 0; j < result->x; j++) {
            float gradientMagnitude = gradientMagnitudes[i * result->x + j];
            unsigned int normalizedMagnitude = (gradientMagnitude > threshold) ? 255 : (unsigned int)(gradientMagnitude / threshold * 255);

            result->data[i * result->x + j].red = normalizedMagnitude;
            result->data[i * result->x + j].green = normalizedMagnitude;
            result->data[i * result->x + j].blue = normalizedMagnitude;
        }
    }

    free(grayscaleImg->data);
    free(grayscaleImg);
    free(gradientMagnitudes);
    free(sortedGradientMagnitudes);

    return result;
}


PPMImg *create_grayscale(PPMImg *colorImage) {
    PPMImg *grayImage = (PPMImg *)malloc(sizeof(PPMImg));
    grayImage->x = colorImage->x;
    grayImage->y = colorImage->y;
    grayImage->data = (PPMPixel *)malloc(colorImage->x * colorImage->y * sizeof(PPMPixel));

    for (int i = 0; i < colorImage->x; i++) {
        for (int j = 0; j < colorImage->y; j++) {
            int luminance = 0.299 * colorImage->data[i * colorImage->y + j].red +
                            0.587 * colorImage->data[i * colorImage->y + j].green +
                            0.114 * colorImage->data[i * colorImage->y + j].blue;

            grayImage->data[i * colorImage->y + j].red = luminance;
            grayImage->data[i * colorImage->y + j].green = luminance;
            grayImage->data[i * colorImage->y + j].blue = luminance;
        }
    }
    return grayImage;
}


PPMImg *gaussian_filter(PPMImg *image, int kernel_dimensions) {
    PPMImg *result;

    result = (PPMImg *)malloc(sizeof(PPMImg));
    result->x = image->x;
    result->y = image->y;
    result->data = (PPMPixel *)malloc(image->x * image->y * sizeof(PPMPixel));

	PPMImg *grayscaleImg = create_grayscale(image);

	for (int i = 0; i < grayscaleImg->x; i++) {
		for (int j = 0; j < grayscaleImg->y; j++) {
			result->data[i * grayscaleImg->x + j].red = 0;
			result->data[i * grayscaleImg->x + j].red = 0;
			result->data[i * grayscaleImg->x + j].red = 0;
		}
	}

	int **kernel = create_gaussian_kernel_filter(sigma_constant, kernel_dimensions);

    int half_kernel_width = (kernel_dimensions * 2 + 1) / 2;
    int half_kernel_height = (kernel_dimensions * 2 + 1) / 2;
    int sumy = 0;
    int sumx = 0;

    for (int i = 0; i < grayscaleImg->y; i++) {
        for (int j = 0; j < grayscaleImg->x; j++) {
          	int redSum = 0, blueSum = 0, greenSum = 0;
    			int normalizationFactor = 0;

          	for (int m = -half_kernel_height; m <= half_kernel_height; m++) {
          	    for (int n = -half_kernel_width; n <= half_kernel_width; n++) {
          	        if (i + m >= 0 && i + m < grayscaleImg->y && j + n >= 0 && j + n < grayscaleImg->x) {
          	            redSum += grayscaleImg->data[(i + m) * grayscaleImg->x + (j + n)].red * kernel[m + half_kernel_height][n + half_kernel_width];
          	            blueSum += grayscaleImg->data[(i + m) * grayscaleImg->x + (j + n)].blue * kernel[m + half_kernel_height][n + half_kernel_width];
          	            greenSum += grayscaleImg->data[(i + m) * grayscaleImg->x + (j + n)].green * kernel[m + half_kernel_height][n + half_kernel_width];
          	            normalizationFactor += kernel[m + half_kernel_height][n + half_kernel_width];
          	        }
          	    }
          	}

		     if (normalizationFactor != 0) {
		     	result->data[i * result->x + j].red = redSum / normalizationFactor;
		     	result->data[i * result->x + j].blue = blueSum / normalizationFactor;
		     	result->data[i * result->x + j].green = greenSum / normalizationFactor;
		  	} else {
		     	result->data[i * grayscaleImg->x + j] = grayscaleImg->data[i * grayscaleImg->x + j];
		     	if (sumx < i) sumx = i;
		     	if (sumy < j) sumy = i;
		  	}

        }
    }
	for (int i = 0; i < 2 * kernel_size + 1; i++) {
        free(kernel[i]);
    }
    free(kernel);

	free(grayscaleImg->data);
	free(grayscaleImg);

    return result;
}


int **create_gaussian_kernel_filter(int sigma, int size) {
    int **kernel = (int **)malloc((2 * size + 1) * sizeof(int *));
    for (int i = 1; i <= 2 * size + 1; i++) {
        kernel[i - 1] = (int *)malloc((2 * size + 1) * sizeof(int));
        for (int j = 1; j <= 2 * size + 1; j++) {
            kernel[i - 1][j - 1] = 159 * (1.0 / (2 * M_PI * pow(sigma, 2))) * exp((-1 * (pow(i - (size + 1), 2) + pow(j - (size + 1), 2)) / (2 * pow(sigma, 2))));
        }
    }
    return kernel;
}


static PPMImg *read_image(const char *filename) {
	char buffer[16];
	PPMImg *image;
	FILE *fp;
	int c;
	int rgb_color;

	fp = fopen(filename, "rb");
	if (!fp) {
		fprintf(stderr, "Unable to open file.");
		exit(1);
	}

	if (!fgets(buffer, sizeof(buffer), fp)) {
		perror(filename);
		exit(1);
	}

	if (buffer[0] != 'P' || buffer[1] != '6') {
		fprintf(stderr, "Image not P6 format.");
		exit(1);
	}

	image = (PPMImg *)malloc(sizeof(PPMImg));
	if (!image) {
		fprintf(stderr, "Unable to allocate memory.");
		exit(1);
	}

	c = getc(fp);
	while(c == '#') {
		while (getc(fp) != '\n');
		c = getc(fp);
	}

	ungetc(c, fp);

    if (fscanf(fp, "%d %d", &image->x, &image->y) != 2) {
         fprintf(stderr, "Invalid image size\n");
         exit(1);
    }

	if (fscanf(fp, "%d", &rgb_color) != 1) {
		fprintf(stderr, "Invalid rgb component");
		exit(1);
	}

	if (rgb_color != RGB_VALUE) {
		fprintf(stderr, "No 8-bit components.");
		exit(1);
	}

	while (fgetc(fp) != '\n');

	image->data = (PPMPixel *)malloc(image->x * image->y * sizeof(PPMPixel));

	if (fread(image->data, 3 * image->x, image->y, fp) != image->y) {
		fprintf(stderr, "Error loading image.");
		exit(1);
	}

	fclose(fp);
	return image;
}


void write_image_to_file(const char *filename, PPMImg *img) {

	FILE *fp;
	fp = fopen(filename, "wb");
	if (!fp) {
		fprintf(stderr, "unable to create image.");
        perror("Error");
	}

	fprintf(fp, "P6\n");

	fprintf(fp, "%d %d\n", img->x, img->y);

	fprintf(fp, "%d\n", RGB_VALUE);

	fwrite(img->data, 3 * img->x, img->y, fp);
	fclose(fp);
}


int main(int argc, char **argv) {
    PPMImg *image;
    image = read_image(argv[1]);

    PPMImg *grayscale_result = create_grayscale(image);

    PPMImg *gaussian_result = gaussian_filter(image, kernel_size);

    PPMImg *sobel_result = sobel_gradient(gaussian_result);

    PPMImg *grad_orientation = gradient_orientation(grayscale_result);

    PPMImg *non_max_suppression = non_maximum_suppression(sobel_result, grad_orientation);

    PPMImg *double_th = double_threshold(non_max_suppression, double_high_treshold , double_low_treshold);

    PPMImg *hysteresis_result = hysteresis(double_th);
    write_image_to_file("canny_edge.ppm", hysteresis_result);

    free(image);

    return 0;
}

