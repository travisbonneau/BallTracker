import cv2 as cv
import numpy as np


def getMidPoint(contour):
    M = cv.moments(contour)
    x = int(M["m10"] / (M["m00"] + 1E-5))
    y = int(M["m01"] / (M["m00"] + 1E-5))
    return x, y


def hsvChange(data):
    global low, high
    lowH = cv.getTrackbarPos("Low H", "HSV Range Finder")
    highH = cv.getTrackbarPos("High H", "HSV Range Finder")
    lowS = cv.getTrackbarPos("Low S", "HSV Range Finder")
    highS = cv.getTrackbarPos("High S", "HSV Range Finder")
    lowV = cv.getTrackbarPos("Low V", "HSV Range Finder")
    highV = cv.getTrackbarPos("High V", "HSV Range Finder")

    low = np.array([lowH, lowS, lowV], np.uint8)
    high = np.array([highH, highS, highV], np.uint8)


cap = cv.VideoCapture(0)
cv.namedWindow("HSV Range Finder", cv.WINDOW_AUTOSIZE)
ret, img = cap.read()
cv.imshow("HSV Range Finder", img)

cv.createTrackbar("Low H", "HSV Range Finder", 0, 180, hsvChange)
cv.createTrackbar("High H", "HSV Range Finder", 180, 180, hsvChange)
cv.createTrackbar("Low S", "HSV Range Finder", 0, 255, hsvChange)
cv.createTrackbar("High S", "HSV Range Finder", 255, 255, hsvChange)
cv.createTrackbar("Low V", "HSV Range Finder", 0, 255, hsvChange)
cv.createTrackbar("High V", "HSV Range Finder", 255, 255, hsvChange)
ret = True
low = np.array([0, 0, 0], np.uint8)
high = np.array([180, 255, 255], np.uint8)

while ret:
    ret, img = cap.read()
    img = cv.flip(img, 1)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    thresh = cv.inRange(hsv, low, high)
    thresh = cv.dilate(thresh, np.ones((7, 7), np.uint8))
    rangeImg = cv.bitwise_and(img, img, mask=thresh)

    cv.imshow("HSV Range Finder", rangeImg)
    cv.imshow("Threshold", thresh)
    cv.imshow("Image", img)
    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()