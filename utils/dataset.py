import csv
import numpy

import numpy as np
import os

import cv2


def readGlands(filesList, datasetPath: str):
    glandsList = []
    imageNameList = []

    # generate samples
    for fileName in filesList:
        img = cv2.imread(os.path.join(datasetPath, fileName))
        try:
            img = cv2.resize(img, (800, 800))
        except:
            print(fileName, "fail")
            pass

        if img is None:
            continue

        imageNameList += [fileName]

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

    return imageNameList, glandsList


def resizeGland(gland, coeffX, coeffY):
    newGland = []
    for point in gland:
        newGland += [list((point[0] * coeffX, point[1] * coeffY))]
    return newGland


def drawGland(img, glands, color=255, stroke=-1):
    for gland in glands:
        nparr = np.array(gland).astype(numpy.int32)
        cv2.drawContours(img, [nparr], 0, color, stroke)
    return img
