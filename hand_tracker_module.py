from sre_constants import SUCCESS
import cv2
import mediapipe as mp
import time

class hand_tracker():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, frame, draw=True):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.mpHands.HAND_CONNECTIONS)
        return frame

    def  findPosition(self, frame, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            target_hand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(target_hand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 8, (139, 0, 0), cv2.FILLED)
        return self.lmList

    def fingersUp(self):
        fingers = []
        # thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
        # fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] -2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = hand_tracker()

    while True:
        SUCCESS, frame = cap.read()
        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame)

        cTime = time.time()
        fps = 1 / (cTime-pTime)
        pTime = cTime
        
        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (0, 255, 0), 3)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()