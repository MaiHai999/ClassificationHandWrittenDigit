from customAlgorithm.ANN import *
from customAlgorithm.PCA import PCA
from customAlgorithm.HistogramEqualization import Histogram_Equalization
from customAlgorithm.Sobel import Sobel
from customAlgorithm.OtusThreshould import OtsuMethod
from customAlgorithm.GaussianFilter import GaussianFilter
from customAlgorithm.DeleteBackground import DeleteBackground

import glob
import numpy as np
import cv2

file_paths1 = ["dataset/0" , "dataset/1" , "dataset/2" , "dataset/3" , "dataset/4" , "dataset/5" , "dataset/6" , "dataset/7" , "dataset/8" , "dataset/9"]

def Processing_Data(file_paths , dsize):
    img_arr = []
    lables_arr = []
    for i,file_path in enumerate(file_paths):
        images = np.array([Processing_Img(file , dsize) for file in glob.glob(file_path+"/*.png")[:10]])
        img_arr.append(images)

        lables = np.array([ Processing_Lable(len(file_paths) , i) for _ in range(len(images))])
        lables_arr.append(lables)

    return np.concatenate(img_arr) , np.concatenate(lables_arr)

def Processing_Img(file , dsize):
    img = cv2.imread(file , 0)
    #delete background
    delte = DeleteBackground(img)
    del_img = delte.remove_white_background()
    #gaussian filter
    gaussian_filter_obj = GaussianFilter(kernel_size=5, sigma=0.7)
    image_filtered = gaussian_filter_obj.apply_filter(del_img)
    #histogram equalization
    hist = Histogram_Equalization(image_filtered)
    image_HE = hist.histogram_equalization()
    #sobel
    sobel = Sobel()
    img_sobel = sobel.apply(image_HE)
    #otsu
    otsu = OtsuMethod(img_sobel)
    thresholded_image = otsu.apply_threshold()

    result_image = np.expand_dims(cv2.resize(thresholded_image.astype(np.uint8), dsize).reshape(-1), axis=0)
    return result_image

def Processing_Lable(lsize , i):
    lable = np.zeros(lsize)
    lable[i] = 1
    lables = np.expand_dims(lable, axis=0)
    return lables

if __name__ == "__main__":
    X_train,y_train = Processing_Data(file_paths1 , (25,25))

    pca = PCA(500)
    X_train = pca.fit_transform(X_train)

    nn = NeuralNetwork()
    nn.add_layer(Layer((1,500) , (1,255)))
    nn.add_layer(ActiveLayer((1,255),(1,255) , ActivationFunctions.sigmoid , ActivationFunctions.sigmoid_prime))
    nn.add_layer(Layer((1,255) , (1,124)))
    nn.add_layer(ActiveLayer((1,124),(1,124) , ActivationFunctions.sigmoid , ActivationFunctions.sigmoid_prime))
    nn.add_layer(Layer((1, 124), (1, 50)))
    nn.add_layer(ActiveLayer((1, 50), (1, 50), ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_prime))
    nn.add_layer(Layer((1, 50), (1, 10)))
    nn.add_layer(ActiveLayer((1, 10), (1, 10), ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_prime))

    nn.set_loss_function(ActivationFunctions.mean_squared_error , ActivationFunctions.mean_squared_error_prime)

    nn.train(X_train , y_train , 0.1 , 100)


    a = Processing_Img("dataset/7/3.png" , (25,25))
    out = nn.predict(pca.transform(a))
    print(out)














