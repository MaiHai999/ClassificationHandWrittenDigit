import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

class GaussianFilter:
    def __init__(x, kernel_size=3, sigma=1.0):
        x.kernel_size = kernel_size
        x.sigma = sigma
        x.kernel = x.gaus_kernel()

    def gaus_kernel(self):
        kernel_gaus = np.fromfunction(
            lambda x, y: (1 / (2 * np.pi * self.sigma ** 2)) * np.exp(
                -((x - (self.kernel_size - 1) / 2) ** 2 + (y - (self.kernel_size - 1) / 2) ** 2) / (2 * self.sigma ** 2)),
            (self.kernel_size, self.kernel_size)
        )
        return kernel_gaus / np.sum(kernel_gaus)

    def apply_filter(self, image):
        image = np.float32(image) / 255.0
        result = cv2.filter2D(image, -1, self.kernel)
        image_filtered = (result * 255).astype(np.uint8)
        return image_filtered

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    path = project_root + "/dataset/sampleImage/bay.jpg"
    image = cv2.imread(path)

    # Tạo đối tượng lọc Gaussian
    gaussian_filter_obj = GaussianFilter(kernel_size=5, sigma=0.7)
    image_filtered = gaussian_filter_obj.apply_filter(image)

    # Hiển thị ảnh gốc và ảnh đã xử lý
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original image')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(image_filtered, cv2.COLOR_BGR2RGB))
    plt.title('After Gaussian filter')

    plt.show()
