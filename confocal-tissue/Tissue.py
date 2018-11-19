import os
from shutil import rmtree

import cv2
import numpy as np

from utils import dataset
from utils.dataset import drawGland, resizeGland

pathToImages = "../dataset/tissue/"
test = pathToImages + "1 (19).jpg"

RESULTS_PATH = "./results/"


# cv2.destroyAllWindows()

def detectGlands(img, imgName, attemptCount):
    # cv2.imshow("src", img)

    oimg = np.copy(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    img = v
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    # cv2.imshow("chache", img)

    thresh, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    # img = cv2.adaptiveThreshold(img, 100, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 0)
    #cv2.imshow("thresh", img)

    binMat = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), None, None,
                              iterations=27)
    #cv2.imshow("bin", binMat)
    cv2.waitKey(0)

    cv2.rectangle(binMat, (0, 0), (np.size(binMat, 0), np.size(binMat, 1)), 255, 10)

    image, contours, hierarchy = cv2.findContours(binMat, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    filteredContours = []

    for contour in contours:

        contourArea = cv2.contourArea(contour)
        if contourArea < attemptCount * 10 * 100 or contourArea > 0.9 * np.size(image, 0) * np.size(image, 1):
            continue

        contourBin = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
        cv2.drawContours(contourBin, [contour], -1, 255.0, 1)
        contourBin = cv2.dilate(contourBin, cv2.getStructuringElement(cv2.MORPH_RECT, (55, 55)))
        contourBin = cv2.bitwise_and(binMat, contourBin, None)

        intersectsBin = cv2.bitwise_and(img, contourBin, None)
        # cv2.imshow("intersects", intersectsBin)

        contourArea = cv2.countNonZero(contourBin)
        intersectsArea = cv2.countNonZero(intersectsBin)

        if intersectsArea < contourArea * 0.40:
            continue

        cv2.drawContours(oimg, [contour], -1, (0.0, 0.0, 255.0), 4)
        filteredContours += [contour]

    # cv2.imshow("glands", oimg)

    return filteredContours, oimg


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
    # shuffle(imageNames)
    # imageNames = imageNames[0:40]

    imageNameList, glandsList = dataset.readGlands(imageNames, pathToImages)

    imageCount = int(len(imageNameList) / 2)

    IMAGE_SIZE = 800

    rmtree(RESULTS_PATH)
    os.mkdir(RESULTS_PATH)

    successCount = 0
    for fileNum in range(0, imageCount):

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

        labelledInnerGlands = list(map(
            lambda glandPair: resizeGland(glandPair[0], float(IMAGE_SIZE) / srcImage.shape[0],
                                          float(IMAGE_SIZE) / srcImage.shape[1]), glands))

        # for attemptCount in range(0, 100):

        detectedInnerContours, oimg = detectGlands(image, imgName, 0)

        drawGland(oimg, labelledInnerGlands, (0.0, 255.0, 0), 2)

        cv2.imshow("glands", oimg)

        percentage = calcPercentage(image, labelledInnerGlands, detectedInnerContours)

        print("ImageNumber: ", fileNum, "percentage:", percentage)

        globalSum += percentage
        if percentage < 33:
            successCount += 1

        # cv2.waitKey(0)

        cv2.imwrite(RESULTS_PATH + imgName, oimg)

        globalCount += 1

    print("Final percentage: ", globalSum / float(globalCount))
    print("Success percentage > 70%: ", successCount / float(globalCount) * 100)


main()
