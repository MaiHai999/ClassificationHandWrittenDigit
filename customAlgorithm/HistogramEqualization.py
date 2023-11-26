import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class Histogram_Equalization:
    def __init__(self, img, bins = 256, ranges = [0,256]):
        self.img = img
        self.bins = bins
        self.ranges = ranges

    def histogram(self):
        histogram = np.zeros(self.bins)
        bin_width = (self.ranges[1] - self.ranges[0]) / self.bins
        for pixel_value in self.img.ravel():
            if self.ranges[0] <= pixel_value <= self.ranges[1]:
                bin_index = int((pixel_value - self.ranges[0]) / bin_width)
                histogram[bin_index] += 1

        return histogram

    def histogram_equalization(self):
        hist = self.histogram()
        Pr = hist / len(self.img.ravel())
        cdf = np.cumsum(Pr)
        s_k = ((self.bins-1) * cdf).astype("uint8")
        equalized_img = cv2.LUT(self.img, s_k)
        return equalized_img

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    path = project_root + "/dataset/sampleImage/bay.jpg"

    img = cv2.imread(path, 0)

    hist = Histogram_Equalization(img)
    s_k = hist.histogram_equalization()

    #show img
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original image')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(s_k, cv2.COLOR_BGR2RGB))
    plt.title('After histogram equalization')

    plt.show()



