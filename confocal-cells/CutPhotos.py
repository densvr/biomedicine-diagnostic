import os, errno
import csv
import cv2

def openGoodPhotos():
    dir_path_pos = "/home/slobodanka/Documents/masterThesis/magisterski/Especially_for_you/+"
    dictPhotoData = {}
    for photo in os.listdir(dir_path_pos):
        dictPhotoData[photo] = []

    with open(dir_path_pos + "/+.csv", encoding="ISO-8859-1") as csvfile:
        cellRows = csv.reader(csvfile)
        for row in cellRows:
            newRow = (str(row[0]).split(";"))
            if newRow[0] in dictPhotoData.keys():
                dictPhotoData[newRow[0]].append(newRow[1:5])

    print(dictPhotoData)
    for photo in dictPhotoData:
        img = cv2.imread(dir_path_pos + "/" + photo)
        # sizeX = int(dictPhotoData.get(photo)[0][0])
        # sizeY = int(dictPhotoData.get(photo)[0][1])
        # print(sizeX, sizeY)
        # img = cv2.resize(img, (sizeX, sizeY))
        for item in dictPhotoData[photo]:
            sizeX = int(item[0])
            sizeY = int(item[1])
            #print(sizeX, sizeY)
            img = cv2.resize(img, (sizeX, sizeY))

            x = int(item[2])
            y = int(item[3])
            crop_img = img[y - 25:y + 50, x - 25:x + 25]

            # try:
            #     cv2.imshow("cropped", crop_img)
            # except:
            #     print('ERROR:', str(photo), sizeX, sizeY)
            # cv2.waitKey(0)


            # img = cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
            # cv2.imshow('aaa', img)
            # cv2.waitKey(0)
           # print("/home/slobodanka/Documents/masterThesis/magisterski/Especially_for_you/cropped_+"+("_".join([str(photo).split(".")[0], str(x), str(y)]))+".jpg")
            cv2.imwrite("/home/slobodanka/Documents/masterThesis/magisterski/Especially_for_you/cropped_+/"+("_".join([str(photo).split(".")[0], str(x), str(y)]))+".jpg", crop_img)

def openBadPhotos():
    dir_path_neg = "/home/slobodanka/Documents/masterThesis/magisterski/Especially_for_you/-"
    directory = "/home/slobodanka/Documents/masterThesis/magisterski/Especially_for_you/cropped_-/"
    dictPhotoData = {}

    for photo in os.listdir(dir_path_neg):
        dictPhotoData[photo] = []

    with open(dir_path_neg + "/table.csv", encoding="ISO-8859-1") as csvfile:
        cellRows = csv.reader(csvfile)
        for row in cellRows:
            newRow = (str(row[0]).split(";"))
            if newRow[0] in dictPhotoData.keys():
                dictPhotoData[newRow[0]].append(newRow[1:5])

    for photo in dictPhotoData:
        img = cv2.imread(dir_path_neg + "/" + photo)

        for item in dictPhotoData[photo]:
            sizeX = int(item[0])
            sizeY = int(item[1])

            img = cv2.resize(img, (sizeX, sizeY))

            x = int(item[2])
            y = int(item[3])
            crop_img = img[y - 25:y + 50, x - 25:x + 25]

            # try:
            #     cv2.imshow("cropped", crop_img)
            # except:
            #     print('ERROR:', str(photo), sizeX, sizeY)
            # cv2.waitKey(0)

            try:
                os.makedirs(directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

            cv2.imwrite(directory + ("_".join([str(photo).split(".")[0], str(x), str(y)]))+".jpg", crop_img)


openGoodPhotos()
# openBadPhotos()