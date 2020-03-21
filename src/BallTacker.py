import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
ret = True
low = np.array([23, 0, 0], np.uint8)
high = np.array([37, 255, 255], np.uint8)
kernel = np.ones((5, 5), np.uint8)
pointQueue = []
frameNum = 0
NUM_FRAMES_TRACKED = 50
showThresh = False


while ret:
    ret, img = cap.read()
    if not ret:
        print("Could Not Get Image")
        break
    img = cv.flip(img, 1)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    thresh = cv.inRange(hsv, low, high)
    thresh = cv.erode(thresh, kernel)
    thresh = cv.dilate(thresh, kernel, iterations=2)
    rangeImg = cv.bitwise_and(img, img, mask=thresh)

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    thresh = cv.inRange(hsv, low, high)
    thresh = cv.erode(thresh, kernel)
    thresh = cv.dilate(thresh, kernel, iterations=2)
    rangeImg = cv.bitwise_and(img, img, mask=thresh)

    cont, heir = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    maxCont = None
    maxArea = 0
    for c in cont:
        area = cv.contourArea(c)
        if area > maxArea:
            maxCont = c
            maxArea = area

    if maxArea < 500:
        maxCont = None

    if maxCont is not None:
        # cv.drawContours(rangeImg, [maxCont], -1, (0, 255, 0))
        (x, y), radius = cv.minEnclosingCircle(maxCont)
        center = (int(x), int(y))
        radius = int(radius)
        cv.circle(img, center, 2, (0, 255, 0), 2)
        cv.circle(img, center, radius, (0, 255, 0), 2)
        pointQueue.append((center, frameNum))

    if len(pointQueue) > 0 and frameNum - pointQueue[0][1] > NUM_FRAMES_TRACKED:
        pointQueue.pop(0)

    frameNum += 1

    for i in range(1, len(pointQueue)):
        if pointQueue[i][1] - pointQueue[i - 1][1] < 2:
            cv.line(img, pointQueue[i - 1][0], pointQueue[i][0], (0, 0, 255))

    if showThresh:
        cv.imshow("Masked Image", rangeImg)
        cv.imshow("Threshold", thresh)
    else:
        cv.destroyWindow("Masked Image")
        cv.destroyWindow("Threshold")

    cv.imshow("Ball Tracker", img)
    key = cv.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('t'):
        showThresh = not showThresh

cv.destroyAllWindows()