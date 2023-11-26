import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

class Sobel:
    def __init__(self):
        self.filterY = np.array([[1.0, 2.0, 1.0],[0.0, 0.0, 0.0],[-1.0, -2.0, -1.0]])
        self.filterX = np.array([[1.0, 0.0, -1.0],[2.0, 0.0, -2.0],[1.0, 0.0, -1.0]])

    def __gradient(self , img):
        sobel_filtered_image = np.zeros_like(img)
        [rows, columns] = np.shape(img)
        for i in range(rows - 2):
            for j in range(columns - 2):
                gx = np.sum(np.multiply(self.filterX, img[i:i + 3, j:j + 3]))
                gy = np.sum(np.multiply(self.filterY, img[i:i + 3, j:j + 3]))
                sobel_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)

        return sobel_filtered_image.astype(np.uint8)
    def apply(self , img):
        imgGradient = self.__gradient(img)
        return imgGradient


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    path = project_root + "/dataset/sampleImage/Lenna.png"
    image = cv2.imread(path , 0)

    # apply soble
    sobel = Sobel()
    img = sobel.apply(image)

    #show img
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original image')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('After sobel filter')

    plt.show()



