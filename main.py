import cv2
from cvzone import HandTrackingModule
import os
import mediapipe as mp
folderPath='frames'
mylist = os.listdir(folderPath)
graphic = [cv2.imread(f'{folderPath}/{imPath}') for imPath in mylist]
intro =graphic[0];
kill =graphic[1];
winner = graphic[2];
cam = cv2.VideoCapture(0)
detector = HandTrackingModule.HandDetector(maxHands=1,detectionCon=0.77)
cv2.imshow('Squid Game', cv2.resize(intro, (0, 0), fx=0.69, fy=0.69))
cv2.waitKey(1)
while True:
    cv2.imshow('Squid Game', cv2.resize(intro, (0, 0), fx=0.69, fy=0.69))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap = cv2.VideoCapture(0)
frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
sqr_img = cv2.imread('img/sqr(2).png')
mlsa =  cv2.imread('img/mlsa.png')
gameOver = False
NotWon =True
capture = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)
mp_Hands = mp.solutions.hands
hands = mp_Hands.Hands()
mpDraw = mp.solutions.drawing_utils
finger_Coord = [(8, 6), (12, 10), (16, 14), (20, 18)]
thumb_Coord = (4,2)
import time
TIMER = int(20)
cap = cv2.VideoCapture(0)  
while True:
    ret, img = cap.read()
    cv2.imshow('a', img)
    k = cv2.waitKey(125)
    if k == ord('t'):
        prev = time.time()
        while TIMER >= 0:
            ret, img = cap.read()
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(TIMER), 
                        (200, 250), font,
                        7, (0, 255, 255),
                        4, cv2.LINE_AA)
            cv2.imshow('a', img)
            cv2.waitKey(125)
            cur = time.time()
            if cur-prev >= 1:
                prev = cur
                TIMER = TIMER-1
  
        else:
            ret, img = cap.read()
            cv2.imshow('a', img)
            cv2.waitKey(200)
            cv2.imwrite('camera.jpg', img)
    elif k == 27:
        break
cap.release()
cv2.destroyAllWindows()
detector = HandTrackingModule.HandDetector(maxHands=2, detectionCon=0.77)
while True:
    isTrue, frame = capture.read()
    hands, img = detector.findHands(frame, flipType=True)

    if hands:
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  
        bbox1 = hand1["bbox"] 
        centerPoint1 = hand1['center']  
        handType1 = hand1["type"]  

        fingers1 = detector.fingersUp(hand1)
        if len(hands) == 2:
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  
            bbox2 = hand2["bbox"]  
            centerPoint2 = hand2['center']  
            handType2 = hand2["type"]  

            fingers2 = detector.fingersUp(hand2)

    cv2.imshow('Video', frame)
    if(cv2.waitKey(20) & 0xFF==ord('q')):
        break
capture.release()
cv2.destroyAllWindows()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)
cv2.imshow('image', sqr_img)
cv2.waitKey(1000)
while not gameOver:
        continue
if NotWon:
    for i in range(10):
       cv2.imshow('Squid Game', cv2.resize(kill, (0, 0), fx=0.69, fy=0.69))
    while True:
        cv2.imshow('Squid Game', cv2.resize(kill, (0, 0), fx=0.69, fy=0.69))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

else:
    cv2.imshow('Squid Game', cv2.resize(winner, (0, 0), fx=0.69, fy=0.69))
    cv2.waitKey(125)

    while True:
        cv2.imshow('Squid Game', cv2.resize(winner, (0, 0), fx=0.69, fy=0.69))
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()
