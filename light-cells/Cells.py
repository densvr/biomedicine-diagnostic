import os

import cv2
import numpy as np
from scipy import ndimage

dirname = '../dataset/light_cells/ER_sheika'
# dirname = 'KM_zheleza'
pictures = [dirname + '/' + name for name in os.listdir(dirname)]

NUMBER = 0
SIZE = 4


# image resizing to the max height of 512 pixels
def resize(image):
    height, width = image.shape[:2]
    max_height = 512
    # max_width = 300

    # only shrink if img is bigger than required
    if max_height < height:  # or max_width < width:
        # get scaling factor
        scaling_factor = max_height / float(height)
        # if max_width / float(width) < scaling_factor:
        # scaling_factor = max_width / float(width)
        # resize image
        return cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)


def labeling(temp_color, temp_binary):
    simage, contours, hierarchy = cv2.findContours(temp_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cell_count = 0
    error_count = 0
    for cnt, hie in zip(contours, hierarchy[0]):
        contourArea = cv2.contourArea(cnt)
        if contourArea > temp_color.shape[0] / 150 * temp_color.shape[1] / 150:
            cv2.drawContours(temp_color, [cnt], 0, (0, 255, 0), 2)
            cell_count += 1
        else:
            cv2.drawContours(temp_color, [cnt], 0, (0, 0, 255), 2)
            error_count += 1

    cv2.putText(temp_color, 'Cells: ' + str(cell_count), (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2,
                cv2.LINE_AA)
    cv2.putText(temp_color, 'Errors: ' + str(error_count), (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2,
                cv2.LINE_AA)
    cv2.imshow('labeled', temp_color)
    return temp_color


def moja_funkcija(image):
    mat = image.copy().astype(float)
    out1 = ndimage.gaussian_laplace(mat, sigma=3)
    out2 = ndimage.gaussian_laplace(mat, sigma=6)
    out3 = ndimage.gaussian_laplace(mat, sigma=9)
    norm_image = cv2.normalize(out1 + out2 + out3, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    ret, otsu = cv2.threshold(norm_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu


def intensity_equalization(value, thresh=1):
    hist, bins = np.histogram(value.ravel(), 256, [0, 256])
    chist = [hist[0]]
    for i in range(1, len(hist)):
        sum = 0
        for j in range(i):
            sum += hist[j]
        chist.append(sum)
    index = 0
    val = chist[len(chist) - 1] * (100 - thresh) / 100
    for k in range(len(chist)):
        if chist[k] > val:
            index = k
            break
    value[value > index] = index
    return value


def moje(image, grayscale, binary, border=True):
    xstep = int(temporary.shape[1] / SIZE)
    ystep = int(temporary.shape[0] / SIZE)

    mat = binary.copy()

    if border:
        image = cv2.copyMakeBorder(image, ystep, ystep, xstep, xstep, cv2.BORDER_REFLECT)
        grayscale = cv2.copyMakeBorder(grayscale, ystep, ystep, xstep, xstep, cv2.BORDER_REFLECT)
        binary = cv2.copyMakeBorder(binary, ystep, ystep, xstep, xstep, cv2.BORDER_REFLECT)

    simage, contours, hierarchy = cv2.findContours(mat, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt, hie in zip(contours, hierarchy[0]):
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            if border:
                cx = cx + xstep
                cy = cy + ystep

            cv2.circle(image, (cx, cy), 2, (0, 0, 255), -1)

            th1 = None
            th2 = None
            th3 = None
            th4 = None
            th5 = None
            # th1 = moja_funkcija(grayscale[cy-ystep:cy,cx-xstep:cx])
            # th2 = moja_funkcija(grayscale[cy:cy + ystep, cx - xstep:cx])
            # th3 = moja_funkcija(grayscale[cy - ystep:cy, cx:cx + xstep])
            # th4 = moja_funkcija(grayscale[cy:cy + ystep, cx:cx + xstep])
            th5 = moja_funkcija(grayscale[cy - (ystep / 2):cy + (ystep / 2), cx - (xstep / 2):cx + (xstep / 2)])

            if th1 is not None:
                binary[cy - ystep:cy, cx - xstep:cx] = cv2.bitwise_or(binary[cy - ystep:cy, cx - xstep:cx],
                                                                      cv2.bitwise_not(moja_funkcija(
                                                                          grayscale[cy - ystep:cy, cx - xstep:cx])))

            if th2 is not None:
                binary[cy:cy + ystep, cx - xstep:cx] = cv2.bitwise_or(binary[cy:cy + ystep, cx - xstep:cx],
                                                                      cv2.bitwise_not(moja_funkcija(
                                                                          grayscale[cy:cy + ystep, cx - xstep:cx])))

            if th3 is not None:
                binary[cy - ystep:cy, cx:cx + xstep] = cv2.bitwise_or(binary[cy - ystep:cy, cx:cx + xstep],
                                                                      cv2.bitwise_not(moja_funkcija(
                                                                          grayscale[cy - ystep:cy, cx:cx + xstep])))

            if th4 is not None:
                binary[cy:cy + ystep, cx:cx + xstep] = cv2.bitwise_or(binary[cy:cy + ystep, cx:cx + xstep],
                                                                      cv2.bitwise_not(moja_funkcija(
                                                                          grayscale[cy:cy + ystep, cx:cx + xstep])))

            if th5 is not None:
                binary[cy - (ystep / 2):cy + (ystep / 2), cx - (xstep / 2):cx + (xstep / 2)] = cv2.bitwise_or(
                    binary[cy - (ystep / 2):cy + (ystep / 2), cx - (xstep / 2):cx + (xstep / 2)],
                    cv2.bitwise_not(moja_funkcija(
                        grayscale[cy - (ystep / 2):cy + (ystep / 2), cx - (xstep / 2):cx + (xstep / 2)])))

    cv2.imshow("binary", binary[ystep:binary.shape[0] - ystep, xstep:binary.shape[1] - xstep])
    cv2.imshow("image", image[ystep:binary.shape[0] - ystep, xstep:binary.shape[1] - xstep])

    return binary[ystep:binary.shape[0] - ystep, xstep:binary.shape[1] - xstep]


while True:

    # global X, Y, NUMBER

    key = cv2.waitKeyEx(1)  # & 0xFF
    if key == 27:
        quit()
    elif key == ord('a'):
        NUMBER = (NUMBER - 1) % len(pictures)
        print("next left picture:", pictures[NUMBER])
    elif key == ord('d'):
        NUMBER = (NUMBER + 1) % len(pictures)
        print("next right picture:", pictures[NUMBER])

    picture = pictures[NUMBER]

    temporary = cv2.imread(picture, 1)
    temporary = resize(temporary)

    # grayscale = cv2.cvtColor(temporary, cv2.COLOR_BGR2GRAY)
    # lab = cv2.cvtColor(temporary, cv2.COLOR_BGR2LAB)
    # grayscale = lab[:,:,0]
    hsv = cv2.cvtColor(temporary, cv2.COLOR_BGR2HSV)
    grayscale = hsv[:, :, 2]
    # grayscale = cv2.bitwise_not(grayscale)

    cv2.imshow("before_intensity_equalization", grayscale)
    grayscale = intensity_equalization(grayscale, thresh=1)
    cv2.imshow("after_intensity_equalization", grayscale)

    thresholded = moja_funkcija(grayscale.copy())
    # labeling(temporary.copy(), cv2.bitwise_not(thresholded.copy()))

    binary = moje(temporary.copy(), grayscale.copy(), cv2.bitwise_not(thresholded.copy()))

    labeling(temporary.copy(), binary.copy())

    cv2.imshow("temporary", temporary)

    cv2.destroyAllWindows()
