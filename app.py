from flask import Flask, render_template, Response
import cv2
import time
import os
import HandTrackingModule as htm

app =Flask(__name__)

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "FingerImages"
myList = os.listdir(folderPath)
#print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)

#print(len(overlayList))
pTime = 0

detector = htm.handDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]

process_this_frame = True

def gen_frames():
    while True:
        success, img = cap.read()  # read the camera frame
        if not success:
            break
        else:
            img = detector.findHands(img)
            lmList = detector.findPosition(img, draw=False)
            # print(lmList)

            if len(lmList) != 0:
                fingers = []

                # Thumb
                if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

                # 4 Fingers
                for id in range(1, 5):
                    if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # print(fingers)
                totalFingers = fingers.count(1)
                print(totalFingers)

                h, w, c = overlayList[totalFingers - 1].shape
                img[0:h, 0:w] = overlayList[totalFingers - 1]

                cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                            10, (255, 0, 0), 25)


            ret, buffer = cv2.imencode('.jpg', img)
            img = buffer.tobytes()
            yield (b'--img\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=img')
if __name__=='__main__':
    app.run(debug=True)