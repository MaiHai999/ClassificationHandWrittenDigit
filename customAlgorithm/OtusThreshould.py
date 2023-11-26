

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

class OtsuMethod:
    def __init__(self, image):
        self.image = image
        self.histogram = self.compute_histogram(image)
        self.total_pixels = image.size

    def compute_histogram(self, image):
        # B1
        hist = [0] * 256
        for pixel in image.flatten():
            hist[pixel] += 1
        return np.array(hist)

    def compute_threshold(self):
        # B2
        total_sum = np.sum(np.arange(256) * self.histogram)
        background_sum = 0
        max_variance = 0
        optimal_threshold = 0

        background_weight = 0
        foreground_weight = 0

        for threshold in range(256):
            background_weight += self.histogram[threshold]
            if background_weight == 0:
                continue

            foreground_weight = self.total_pixels - background_weight
            if foreground_weight == 0:
                break

            background_sum += threshold * self.histogram[threshold]
            background_mean = background_sum / background_weight
            foreground_mean = (total_sum - background_sum) / foreground_weight

            # B3
            variance_between = background_weight * foreground_weight * (background_mean - foreground_mean) ** 2

            # B4
            if variance_between > max_variance:
                max_variance = variance_between
                optimal_threshold = threshold

        return optimal_threshold

    def apply_threshold(self):
        threshold = self.compute_threshold()
        return (self.image > threshold) * 255

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    path = project_root + "/dataset/sampleImage/Lenna.png"
    image = cv2.imread(path, 0)

    otsu = OtsuMethod(image)
    thresholded_image = otsu.apply_threshold()

    # Hiển thị kết quả
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(thresholded_image)
    plt.title('Otsu Thresholding')
    plt.xticks([]), plt.yticks([])
    plt.show()