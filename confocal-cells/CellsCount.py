import os

import cv2
import numpy as np

# pathToImages = "/home/slobodanka/Documents/masterThesis/CellsProject-master/images/"
pathToImages = "../dataset/cells/cells_new"

dictPhotos = {}
dictPhotos = {
    "1(5)": 11,
    "1(6)": 11,
    "1(7)": 95,
    "1(9)": 12,
    "1(13)": 360,
    "1(30)": 145,
    "1(31)": 128,
    "1(38)": 45,
    "1(41)": 189,
    "1(42)": 123
}


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
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

        if cv2.contourArea(c) < 400:
            continue

        if cX != cY:
            cv2.drawContours(oimg, [c], -1, (0, 255, 0), 2)
            count += 1
    # print('COUNT', count)

    # return oimg
    return count


def process_image(srcImg):
    oimg = np.copy(srcImg)
    hsv = cv2.cvtColor(srcImg, cv2.COLOR_BGR2HSV)
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


    horizontalOutputImage = np.concatenate((srcImg, oimg), axis=1)
    cv2.imshow("src + final", horizontalOutputImage)

    cv2.imshow("final", oimg)

    #k = cv2.waitKey(0)

    return numberOfCells


# cv2.destroyAllWindows()


def calcPercentage(imgName, numberOfCells):
    imageName = imgName.split('.')[0]
    # print("ImageNumber: ", imageNumber, " cells: ", dictPhotos[imageNumber], " observed: ",\
    #      numberOfCells, "percentage:", (100.0 -percentage), "or real:", percentage)
    percentage = min(dictPhotos[imageName], numberOfCells) / max((dictPhotos[imageName], numberOfCells)) * 100
    print("ImageNumber: ", imageName, " cells: ", dictPhotos[imageName], " observed: ", \
          numberOfCells, "percentage:", percentage)
    return percentage


def main():
    globalSum = 0
    globalCount = 0
    dirs = os.listdir(pathToImages)
    for imgName in dirs:
        img = cv2.imread(os.path.join(pathToImages, imgName))

        #try:
        #    img = cv2.resize(img, (450, 450))
        #except:
        #    pass

        #cv2.imshow("src", img)

        if img is not None:
            print(imgName)
            numberOfCells = process_image(img)
            globalSum += calcPercentage(imgName, numberOfCells)
            globalCount += 1

    print("Final percentage: ", globalSum / float(globalCount))


main()
