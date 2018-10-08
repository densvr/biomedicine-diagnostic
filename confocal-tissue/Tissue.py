import os

import cv2
import numpy as np
from random import shuffle

# pathToImages = "/home/slobodanka/Documents/masterThesis/CellsProject-master/images/"
from utils import dataset
from utils.dataset import drawGland, resizeGland

pathToImages = "/Users/danser/Google Drive/post graduate/cell couting on digital microscopy images/projects/biomedicine-diagnostic/dataset/tissue/"
test = pathToImages + "1 (19).jpg"

dictPhotos = {1: 40, 2: 28, 3: 145, 4: 112, 8: 36, 9: 13, 10: 91, 11: 1516, 12: 362, 13: 419, 14: 257, 15: 228, 16: 121,
              17: 136,
              18: 110, 19: 856, 20: 819, 21: 964, 22: 885, 23: 928, 28: 915, 29: 770, 30: 164, 33: 43, 34: 23, 37: 44,
              38: 52, 39: 52,
              40: 147, 43: 92, 44: 63, 45: 136, 46: 97, 47: 113, 52: 44, 53: 42, 54: 50, 55: 41, 56: 35, 57: 159,
              58: 131, 60: 95, 61: 102,
              62: 100, 65: 53, 68: 100, 71: 95, 72: 100, 75: 19, 76: 27, 79: 6, 85: 38, 86: 26, 87: 50, 88: 71, 89: 34,
              90: 35, 91: 26, 92: 37}


def segment(img):
    new_img = np.copy(img)
    new_img = np.float64(new_img)
    pos = new_img > (np.mean(img) + 20)
    new_img[pos] += 100
    others = new_img < (np.mean(img) + 20)
    new_img[new_img > 255] = 255
    new_img[others] -= 100
    new_img[new_img < 0] = 0
    return new_img


def threshold(img):
    img = np.uint8(img)
    return img


def count_cells(img, oimg):
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(oimg, contours, -1, (0, 255, 255), 1)
    # print("The number of cells are", len(contours))

    count = 0
    for c in contours:
        M = cv2.moments(c)
        denom = M['m00']
        if denom == 0:
            denom = 1
        cX = int(M['m10'] / denom)
        cY = int(M['m01'] / denom)

        if cX != cY:
            cv2.drawContours(oimg, [c], -1, (0, 255, 0), 2)
            count += 1
    # print('COUNT', count)

    # return oimg
    return count


def process_image(img):
    oimg = np.copy(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    img = v
    # cv2.imshow("Gray2.tif", img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    img = clahe.apply(img)
    #    cv2.imshow('filtered', img)

    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    #  cv2.imshow('erode', img)

    img = segment(img)
    #   cv2.imshow('segment', img)
    img = threshold(img)
    #    cv2.imshow('threshold', img)

    # img = cv2.GaussianBlur(img, (10,10), 0)
    img = cv2.medianBlur(img, 5)
    #    cv2.imshow('median', img)
    numberOfCells = count_cells(img, oimg)
    # cv2.imshow("final", oimg)
    k = cv2.waitKey(0)

    return numberOfCells


# cv2.destroyAllWindows()

def detectGlands(img):
    # cv2.imshow("src", img)

    oimg = np.copy(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    img = v
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    #cv2.imshow("chache", img)

    # thresh, img = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
    # img = cv2.adaptiveThreshold(img, 100, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 0)
    # cv2.imshow("thresh", img)

    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = segment(img)
    img = threshold(img)
    img = cv2.medianBlur(img, 5)
    thresh, binMat = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(oimg, contours, -1, (0, 255, 255), 1

    # cv2.imshow("bin_before", binMat)

    binMat = cv2.dilate(binMat, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=25)
    binMat = cv2.erode(binMat, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=34)

    # cv2.imshow("bin", binMat)
    # cv2.waitKey(0)

    cv2.rectangle(binMat, (0, 0), (np.size(binMat, 0), np.size(binMat, 1)), 255, 10)

    image, contours, hierarchy = cv2.findContours(binMat, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    filteredContours = []

    for contour in contours:

        contourArea = cv2.contourArea(contour)
        if contourArea < 100 * 100 or contourArea > 0.9 * np.size(image, 0) * np.size(image, 1):
            continue

        cv2.drawContours(oimg, [contour], -1, (0.0, 0.0, 255.0), 2)
        filteredContours += [contour]

    cv2.imshow("glands", oimg)

    return filteredContours


def calcPercentage(image, labelledInnerGlands, detectedInnerContours):
    mask1 = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
    mask2 = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)

    mask1 = drawGland(mask1, labelledInnerGlands, 100)
    mask2 = drawGland(mask2, detectedInnerContours, 100)
    sum = mask1 + mask2

    intersectsImage = np.copy(sum)
    cv2.threshold(intersectsImage, 199, 255, cv2.THRESH_BINARY, intersectsImage)
    intersectsCount = cv2.countNonZero(intersectsImage)

    summaryImage = np.copy(sum)
    cv2.threshold(summaryImage, 99, 255, cv2.THRESH_BINARY, summaryImage)
    summaryCount = cv2.countNonZero(summaryImage)

    if summaryCount == 0:
        return 100
    return intersectsCount / float(summaryCount) * 100.0


def main():
    globalSum = 0
    globalCount = 0
    imageNames = os.listdir(pathToImages)
    imageNames.sort()
    #shuffle(imageNames)
    #imageNames = imageNames[0:40]

    imageNameList, glandsList = dataset.readGlands(imageNames, pathToImages)

    imageCount = int(len(imageNameList) / 2)

    IMAGE_SIZE = 800

    successCount = 0
    for fileNum in range(0, imageCount):
        # if count < 2:
        #    continue
        imgName = imageNameList[fileNum]
        srcImage = cv2.imread(os.path.join(pathToImages, imgName))

        glands = glandsList[fileNum]
        image = None
        try:
            image = cv2.resize(srcImage, (IMAGE_SIZE, IMAGE_SIZE))
        except:
            pass
        if image is None:
            continue

        print(imgName)
        detectedInnerContours = detectGlands(image)

        labelledInnerGlands = list(map(
            lambda glandPair: resizeGland(glandPair[0], float(IMAGE_SIZE) / srcImage.shape[0],
                                          float(IMAGE_SIZE) / srcImage.shape[1]), glands))

        percentage = calcPercentage(image, labelledInnerGlands, detectedInnerContours)
        globalSum += percentage

        if percentage > 50:
            successCount += 1

        print("ImageNumber: ", fileNum, "percentage:", percentage)

        globalCount += 1
        #cv2.waitKey()

    print("Final percentage: ", globalSum / float(globalCount))
    print("Success percentage > 70%: ", successCount / float(globalCount) * 100)


main()
