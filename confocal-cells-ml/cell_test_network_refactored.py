import argparse
import os

import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained model model")
ap.add_argument("-p", "--path", required=True,
                help="folder to input images")
args = vars(ap.parse_args())

count = 0
dict_group = {}
path = args["path"]

pathToImages = "/home/slobodankac/pic/"
FILENAME = "/home/slobodanka/pic/16.jpg"
PHOTONUMBER="16"
HIGH_PARAM = 20
LOW_PARAM = 20


def segment(img):
    new_img = np.zeros(img.shape, dtype=np.float64)
    new_img = np.copy(img)
    new_img = np.float64(new_img)  # So that we can exeed 255
    pos = new_img > (np.mean(img) + HIGH_PARAM)
    new_img[pos] += 100
    others = new_img < (np.mean(img) + LOW_PARAM)
    new_img[new_img > 255] = 255
    new_img[others] -= 100
    new_img[new_img < 0] = 0
    return new_img


def threshold(img):
    img = np.uint8(img)
    return img


def count_cells(img, oimg):
    original = np.copy(img)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)
    blur = cv2.GaussianBlur(cl1, (5, 5), 0)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow("image", image)

    f = open('eratio.csv', 'w')
    f.write('x,y,ratio\n')
    count = 0

    # loop over the contours
    for c in contours:
        # compute the center of the contour
        M = cv2.moments(c)
        denom = M['m00']
        if denom == 0:
            denom = 1

        area = cv2.contourArea(c)
        x, y, width, height = cv2.boundingRect(c)

        if area > 5.0:
            count += 1
            crop_img = original[y:y + height, x:x + width]
            cv2.imwrite(
                '/home/slobodankac/PycharmProjects/cell_counting_CNN/cutted3/' + ("_".join([str(x), str(y)])) + ".jpg",
                crop_img)
            new_img = '/home/slobodankac/PycharmProjects/cell_counting_CNN/cutted3/' + str(
                "_".join([str(x), str(y)]) + ".jpg")
            image = cv2.imread(new_img)

            # pre-process the image for classification
            image = cv2.resize(image, (28, 28))

            image = image.astype("float") / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            # load the trained convolutional neural network

            model = load_model(args["model"])

            aaa = model.predict(image)[0]
            indices = np.where(aaa == aaa.max())  # this is how to get the index of the maximum element

            label = indices[0]
            if indices[0] == 0:
                label = "group1"
            elif indices[0] == 1:
                label = "group2"

            if label == "group1":
                count += 1
                cv2.drawContours(oimg, [c], -1, (0, 255, 0), 2)
            else:
                cv2.drawContours(oimg, [c], -1, (255, 0, 0), 1)
    return oimg


def process_image(img):
    oimg = np.copy(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    img = v
    # cv2.imshow("Gray2.tif", img)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #
    # img = clahe.apply(img)
    # cv2.imshow('filtered', img)
    # img = cv2.GaussianBlur(img, (5, 5), 0)

    img = count_cells(img, oimg)
    cv2.imshow("final", img)

    k = cv2.waitKey(0)
    return 0


# cv2.destroyAllWindows()

dictPhotos = {}
dictPhotos = {1: 40, 2: 28, 3: 145, 4: 112, 8: 36, 9: 13, 10: 91, 11: 1516, 12: 362, 13: 419, 14: 257, 15: 228, 16: 121,
              17: 136,
              18: 110, 19: 856, 20: 819, 21: 964, 22: 885, 23: 928, 28: 915, 29: 770, 30: 164, 33: 43, 34: 23, 37: 44,
              38: 52, 39: 52,
              40: 147, 43: 92, 44: 63, 45: 136, 46: 97, 47: 113, 52: 44, 53: 42, 54: 50, 55: 41, 56: 35, 57: 159,
              58: 131, 60: 95, 61: 102,
              62: 100, 65: 53, 68: 100, 71: 95, 72: 100, 75: 19, 76: 27, 79: 6, 85: 38, 86: 26, 87: 50, 88: 71, 89: 34,
              90: 35, 91: 26, 92: 37}


def calculatePercantge(imgName, numberOfCells):
    imageNumber = int(imgName.split(".")[0])
    percentage = ((abs(dictPhotos[imageNumber] - numberOfCells)) / dictPhotos[imageNumber]) * 100
    print("ImageNumber: ", imageNumber, " cells: ", dictPhotos[imageNumber], " observed: ", \
          numberOfCells, "percentage:", (100.0 - percentage), "or real:", percentage)
    return (100.0 - percentage)


def main():
    globalSum = 0
    globalCount = 0
    for imgName in os.listdir(pathToImages):
        # if "1_endo_R_Mel1A+1B_40x_3" in imgName:
        if PHOTONUMBER in imgName:
            # print("pathToImages", pathToImages, imgName)
            img = cv2.imread(os.path.join(pathToImages, imgName))
            try:
                img = cv2.resize(img, (650, 650))
            except:
                pass
            #
            #  da probam bez resize
            #
            if img is not None:
                print(imgName)
                numberOfCells = process_image(img)
                globalSum += calculatePercantge(imgName, numberOfCells)
                globalCount += 1

    print("Final percentage: ", globalSum / float(globalCount))


main()
