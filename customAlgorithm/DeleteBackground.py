
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt



class DeleteBackground:
    def __init__(self , img):
        self.img = img

    def remove_white_background(self):
        # Tạo ảnh trắng có cùng kích thước với ảnh gốc
        white_image = np.ones_like(self.img) * 255

        # Phép trừ để giữ lại đối tượng và xoá nền màu trắng
        result_image = np.abs(np.subtract(white_image, self.img)  )

        return result_image


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    path = project_root + "/dataset/0/0.png"
    image = cv2.imread(path , 0)

    # delete background
    delte = DeleteBackground(image)
    img = delte.remove_white_background()

    #show img
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original image')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('After delete background')

    plt.show()
