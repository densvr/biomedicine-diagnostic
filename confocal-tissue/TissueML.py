import os

import cv2
import csv
import numpy as np

# pathToImages = "/home/slobodanka/Documents/masterThesis/CellsProject-master/images/"
datasetPath = "/Users/danser/Google Drive/post graduate/cell couting on digital microscopy images/projects/biomedicine-diagnostic/dataset/tissue/"
test = datasetPath + "1 (19).jpg"

dictPhotos = {}
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
    oimg = np.copy(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    img = v
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = segment(img)
    img = threshold(img)
    img = cv2.medianBlur(img, 5)
    thresh, binMat = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(oimg, contours, -1, (0, 255, 255), 1

    binMat = cv2.dilate(binMat, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=22)
    binMat = cv2.erode(binMat, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=33)

    # cv2.imshow("bin", binMat)
    # cv2.waitKey(0)

    cv2.rectangle(binMat, (0, 0), (np.size(binMat, 0), np.size(binMat, 1)), 255, 10)

    image, contours, hierarchy = cv2.findContours(binMat, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:

        contourArea = cv2.contourArea(contour)
        if contourArea < 100 * 100 or contourArea > 0.9 * np.size(image, 0) * np.size(image, 1):
            continue

        cv2.drawContours(oimg, [contour], -1, (0.0, 0.0, 255.0), 2)

    cv2.imshow("glands", oimg)
    cv2.waitKey()


def calcPercentage(imgName, numberOfCells):
    imageNumber = int(imgName.split('.')[0])
    percentage = ((abs(dictPhotos[imageNumber] - numberOfCells)) / dictPhotos[imageNumber]) * 100
    # print("ImageNumber: ", imageNumber, " cells: ", dictPhotos[imageNumber], " observed: ",\
    #      numberOfCells, "percentage:", (100.0 -percentage), "or real:", percentage)
    print("ImageNumber: ", imageNumber, " cells: ", dictPhotos[imageNumber], " observed: ", \
          numberOfCells, "percentage:",
          min(dictPhotos[imageNumber], numberOfCells) / max((dictPhotos[imageNumber], numberOfCells)))
    return min(dictPhotos[imageNumber], numberOfCells) / max((dictPhotos[imageNumber], numberOfCells))


def main():
    globalSum = 0
    globalCount = 0
    for imgName in os.listdir(datasetPath):
        img = cv2.imread(os.path.join(datasetPath, imgName))
        try:
            img = cv2.resize(img, (800, 800))
        except:
            pass

        if img is None:
            continue

        glands = []

        with open(datasetPath + imgName + '.csv', 'rt') as f:
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


        print(imgName)
        detectGlands(img)
        # globalSum += calcPercentage(imgName, numberOfCells)
        globalCount += 1

    print("Final percentage: ", globalSum / float(globalCount))


main()
