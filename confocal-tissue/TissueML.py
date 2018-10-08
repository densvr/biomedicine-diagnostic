import csv
import os
import random

import cv2
import numpy as np
from sklearn import neural_network

from utils.dataset import readGlands

datasetPath = "/Users/danser/Google Drive/post graduate/cell couting on digital microscopy images/projects/biomedicine-diagnostic/dataset/tissue/"


# cv2.destroyAllWindows()


def generateSamples(image, gland, countInside, countOutside, sampleSize, sampleResizedSize, isDebug: bool):
    nparrInside = np.array(gland[0])
    nparrOutside = np.array(gland[1])
    contourMask: np.ndarray = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
    # cv2.drawContours(contourMask, [nparrOutside], 0, color=255, thickness=-1)
    cv2.drawContours(contourMask, [nparrInside], 0, color=255, thickness=-1)
    insideImages = []
    outsideImages = []
    insideRects = []
    outsideRects = []
    while np.size(insideImages, 0) < countInside or np.size(outsideImages, 0) < countOutside:
        (x, y) = random.randint(0, np.size(image, 0) - sampleSize[0] - 1), \
                 random.randint(0, np.size(image, 0) - sampleSize[1] - 1)

        rect = (x, sampleSize[0], y, sampleSize[1])
        croppedImage = cropImage(image, rect)
        croppedImage = cv2.resize(croppedImage, sampleResizedSize)

        imgDraw = image.copy()

        if contourMask[int(y + sampleSize[1] / 2), int(x + sampleSize[0] / 2)] > 0:
            if np.size(insideImages, 0) < countInside:
                insideImages += [croppedImage]
                insideRects += [rect]
                if isDebug:
                    cv2.rectangle(imgDraw, (rect[0], rect[2]), (rect[0] + rect[1], rect[2] + rect[3]), (0, 0, 255), 3)
                    showImage("contour", imgDraw)
                    cv2.imshow("cropped", croppedImage)
                    cv2.waitKey()
        else:
            if np.size(outsideImages, 0) < countOutside:
                outsideImages += [croppedImage]
                outsideRects += [rect]
                if isDebug:
                    cv2.rectangle(imgDraw, (rect[0], rect[2]), (rect[0] + rect[1], rect[2] + rect[3]), (0, 255, 0), 3)
                    showImage("contour", imgDraw)
                    cv2.imshow("cropped", croppedImage)
                    cv2.waitKey()

    return insideImages, outsideImages, insideRects, outsideRects


def generateTestSamples(image, gland, countInside, countOutside, sampleSize, sampleResizedSize, isDebug: bool):
    nparrInside = np.array(gland[0])
    nparrOutside = np.array(gland[1])
    contourMask: np.ndarray = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
    # cv2.drawContours(contourMask, [nparrOutside], 0, color=255, thickness=-1)
    cv2.drawContours(contourMask, [nparrInside], 0, color=255, thickness=-1)
    insideImages = []
    outsideImages = []
    insideRects = []
    outsideRects = []

    for i in range(0, np.size(image, 0) - sampleSize[0] - 1):
        for j in range(0, np.size(image, 1) - sampleSize[1] - 1):
            if i % 50 != 0 or j % 50 != 0:
                continue
            x = i
            y = j

            rect = (x, sampleSize[0], y, sampleSize[1])
            croppedImage = cropImage(image, rect)
            croppedImage = cv2.resize(croppedImage, sampleResizedSize)

            imgDraw = image.copy()

            if contourMask[int(y + sampleSize[1] / 2), int(x + sampleSize[0] / 2)] > 0:
                insideImages += [croppedImage]
                insideRects += [rect]
                if isDebug:
                    cv2.rectangle(imgDraw, (rect[0], rect[2]), (rect[0] + rect[1], rect[2] + rect[3]),
                                  (0, 0, 255), 3)
                    showImage("contour", imgDraw)
                    cv2.imshow("cropped", croppedImage)
                    cv2.waitKey()
            else:
                outsideImages += [croppedImage]
                outsideRects += [rect]
                if isDebug:
                    cv2.rectangle(imgDraw, (rect[0], rect[2]), (rect[0] + rect[1], rect[2] + rect[3]),
                                  (0, 255, 0), 3)
                    showImage("contour", imgDraw)
                    cv2.imshow("cropped", croppedImage)
                    cv2.waitKey()

    return insideImages, outsideImages, insideRects, outsideRects


def flattenSamples(images):
    return list(map(lambda x: flattenImage(x), images))


def cropImage(image, rect):
    return image[
           max(rect[2], 0): min(rect[2] + rect[3], np.size(image, 1)),
           max(rect[0], 0): min(rect[0] + rect[1], np.size(image, 0))]


def flattenImage(image):
    return np.reshape(image, -1)


def showImage(name, img):
    resizedImg = cv2.resize(img, (600, 600))
    cv2.imshow(name, resizedImg)


def main():
    globalSum = 0
    globalCount = 0

    filesList = os.listdir(datasetPath)[0:40]

    imageNameList, glandsList = readGlands(filesList, datasetPath)

    learningImageCount = int(len(imageNameList) / 2)

    for fileIdx in range(0, learningImageCount):
        fileName = imageNameList[fileIdx]
        glands = glandsList[fileIdx]

        srcImg = cv2.imread(os.path.join(datasetPath, fileName))

        # generate images
        curInsideImages, curOutsideImages, curInsideRects, curOutsideRects = \
            generateSamples(srcImg, glands[0], 50, 50, (15, 15), (10, 10), False)
        insideImages += curInsideImages
        outsideImages += curOutsideImages
        insideRects += curInsideRects
        outsideRects += curOutsideRects

    insideSamples = flattenSamples(insideImages)
    outsideSamples = flattenSamples(outsideImages)

    classifier = neural_network.MLPClassifier(hidden_layer_sizes=100)

    classifier.fit(insideSamples + outsideSamples,
                   list(np.ones((np.size(insideSamples, 0)))) + list(np.zeros((np.size(outsideSamples, 0)))))

    for fileIdx in range(learningImageCount + 1, len(imageNameList)):

        fileName = imageNameList[fileIdx]
        glands = glandsList[fileIdx]

        img = cv2.imread(os.path.join(datasetPath, fileName))
        srcImg = img
        try:
            img = cv2.resize(img, (800, 800))
        except:
            pass

        if img is None:
            continue

        imgWithLabelledGlands = srcImg.copy()
        for gland in glands:
            nparr = np.array(gland[0])
            cv2.drawContours(imgWithLabelledGlands, [nparr], 0, (0, 255, 0), 4)
            nparr = np.array(gland[1])
            cv2.drawContours(imgWithLabelledGlands, [nparr], 0, (0, 255, 255), 4)
        showImage(fileName + " labelled glands", imgWithLabelledGlands)
        cv2.waitKey(0)

        while True:

            insideImages, outsideImages, insideRects, outsideRects = generateTestSamples(srcImg, glands[0], 100, 100,
                                                                                         (100, 100), (10, 10), False)
            insideSamples = flattenSamples(insideImages)
            outsideSamples = flattenSamples(outsideImages)

            insidePredictions = classifier.predict(insideSamples)
            outsidePredicitions = classifier.predict(outsideSamples)

            predictions = list(insidePredictions) + list(outsidePredicitions)
            rects = insideRects + outsideRects

            imgDraw = imgWithLabelledGlands.copy()

            imgOverlay = imgDraw.copy()
            count = 0
            for rect in rects:
                if predictions[count] > 0.5:
                    cv2.rectangle(imgOverlay, (rect[0], rect[2]), (rect[0] + rect[1], rect[2] + rect[3]), (0, 255, 0),
                                  -1)
                else:
                    cv2.rectangle(imgOverlay, (rect[0], rect[2]), (rect[0] + rect[1], rect[2] + rect[3]), (0, 0, 255),
                                  -1)
                count += 1

            cv2.addWeighted(imgDraw, 0.8, imgOverlay, 0.2, 0, imgDraw)
            showImage("result", imgDraw)

            k = cv2.waitKey()
            if k == 27:  # Esc key to stop
                break

        print(fileName)
        # detectGlands(img)
        # globalSum += calcPercentage(fileName, numberOfCells)
        globalCount += 1

    print("Final percentage: ", globalSum / float(globalCount))


main()
