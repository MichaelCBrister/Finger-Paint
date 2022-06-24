from sre_constants import SUCCESS
import cv2
from cv2 import LMEDS
import numpy as np
import os
import hand_tracker_module as tr

folderPath = "Header"
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    img = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(img)
header = overlayList[0]
drawColor = (0, 0, 255)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

detector = tr.hand_tracker()
xp, yp = 0, 0
canvas = np.zeros((720, 1280, 3), np.uint8)
while True:
    SUCCESS, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    frame  = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw = False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()

        # select color
        if fingers[1] and fingers[2]:
            xp, yp, = 0, 0
            if y1 < 125:
                # red
                if 50 < x1 <155:
                    header = overlayList[0]
                    drawColor = (0, 0, 255)
                # green
                elif 425 < x1 < 525:
                    header = overlayList[1]
                    drawColor = (0, 255, 0)
                # blue
                elif 750 < x1 < 850:
                    header = overlayList[2]
                    drawColor = (255, 0, 0)
                # eraser
                elif 1150 < x1 < 1280:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(frame, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)
        # draw
        if fingers[1] and fingers[2] == False:
            cv2.circle(frame, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawColor == (0, 0, 0):
                cv2.line(frame, (xp, yp), (x1, y1), drawColor, 40)
                cv2.line(canvas, (xp, yp), (x1, y1), drawColor, 40)
            else:
                cv2.line(frame, (xp, yp), (x1, y1), drawColor, 15)
                cv2.line(canvas, (xp, yp), (x1, y1), drawColor, 15)
            xp, yp = x1, y1

    gry_frame = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, frame_inverse = cv2.threshold(gry_frame, 50, 255, cv2.THRESH_BINARY_INV)
    frame_inverse = cv2.cvtColor(frame_inverse, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, frame_inverse)
    img = cv2.bitwise_or(frame, canvas)

    frame[0:125, 0:1280] = header
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.imshow("Finger Paint", frame)
    if cv2.waitKey(1) == ord('q'):
            break

