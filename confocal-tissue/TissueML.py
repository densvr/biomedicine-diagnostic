import os
import random

import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics, neural_network

datasetPath = "/Users/danser/Google Drive/post graduate/cell couting on digital microscopy images/projects/biomedicine-diagnostic/dataset/tissue/"


# cv2.destroyAllWindows()


def generateSamples(image, gland, countInside, countOutside, sampleSize, sampleResizedSize, isDebug: bool):
    nparr = np.array(gland[0])
    contourMask: np.ndarray = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
    cv2.drawContours(contourMask, [nparr], 0, color=255, thickness=-1)
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

    filesList = os.listdir(datasetPath)

    glandsList = []

    # generate samples
    for fileName in filesList:
        img = cv2.imread(os.path.join(datasetPath, fileName))
        try:
            img = cv2.resize(img, (800, 800))
        except:
            pass

        if img is None:
            continue

        glands = []

        with open(datasetPath + fileName + '.csv', 'rt') as f:
            reader = csv.reader(f)
            gland = []
            for row in reader:
                try:
                    cells = list(map(lambda x: int(x), filter(lambda x: x.strip() != "", row[0].split(";"))))
                    points = []
                    for i in range(1, int(np.size(cells, 0) / 2 - 1)):
                        points += [[cells[i * 2], cells[i * 2 + 1]]]

                    gland += [points]
                    if np.size(gland, 0) == 2:
                        glands += [gland]
                        gland = []

                except:
                    continue

        glandsList += [glands]

    for fileIdx in range(0, len(filesList)):

        fileName = filesList[fileIdx]
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

        # generate images
        insideImages, outsideImages, insideRects, outsideRects = generateSamples(srcImg, glands[0], 500, 500,
                                                                                 (100, 100), (10, 10), False)

        insideSamples = flattenSamples(insideImages)
        outsideSamples = flattenSamples(outsideImages)

        classifier = neural_network.MLPClassifier(hidden_layer_sizes=(100,50))

        classifier.fit(insideSamples + outsideSamples,
                       list(np.ones((np.size(insideSamples, 0)))) + list(np.zeros((np.size(outsideSamples, 0)))))

        while True:

            insideImages, outsideImages, insideRects, outsideRects = generateSamples(srcImg, glands[0], 100, 100,
                                                                                     (100, 100), (10, 10), False)
            insideSamples = flattenSamples(insideImages)
            outsideSamples = flattenSamples(outsideImages)

            insidePredictions = classifier.predict(insideSamples)
            outsidePredicitions = classifier.predict(outsideSamples)

            predictions = list(insidePredictions) + list(outsidePredicitions)
            rects = insideRects + outsideRects

            imgDraw = imgWithLabelledGlands.copy()
            count = 0
            for rect in rects:
                if predictions[count] > 0.5:
                    cv2.rectangle(imgDraw, (rect[0], rect[2]), (rect[0] + rect[1], rect[2] + rect[3]), (0, 255, 0), 2)
                else:
                    cv2.rectangle(imgDraw, (rect[0], rect[2]), (rect[0] + rect[1], rect[2] + rect[3]), (0, 0, 255), 2)
                count += 1

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
