from customAlgorithm.ANN import *
from customAlgorithm.PCA import *

import glob
import numpy as np
import cv2

file_paths1 = ["dataset/0/0" , "dataset/1/1" , "dataset/2/2" , "dataset/3/3" , "dataset/4/4" , "dataset/5/5" , "dataset/6/6" , "dataset/7/7" , "dataset/8/8" , "dataset/9/9"]

def Processing_Data(file_paths , dsize):
    img_arr = []
    lables_arr = []
    for i,file_path in enumerate(file_paths):
        images = np.array([ Processing_Img(file , dsize) for file in glob.glob(file_path+"/*.png")[:10]])
        img_arr.append(images)

        lables = np.array([ Processing_Lable(len(file_paths) , i) for _ in range(len(images))])
        lables_arr.append(lables)

    return np.concatenate(img_arr) , np.concatenate(lables_arr)

def Processing_Img(file , dsize):
    img = cv2.imread(file , cv2.IMREAD_UNCHANGED)
    img = np.expand_dims(cv2.resize(img, dsize).reshape(-1), axis=0)
    return img

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

    nn.train(X_train , y_train , 0.1 , 10)


    a = Processing_Img("dataset/7/7/3.png" , (25,25))
    out = nn.predict(pca.transform(a))
    print(out)




















