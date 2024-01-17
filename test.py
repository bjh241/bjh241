#
# 모듈을 사용하는 Hand Tracking 코드.
#

import cv2
import handTracking as ht

my_cap = cv2.VideoCapture(0)                  # 0 = 첫번째 카메라.
my_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)     # 카메라 입력의 너비.
my_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)    # 카메라 입력의 높이.
my_detector = ht.HandDetector()               # 검출 객체 생성.
while True:
    _, img = my_cap.read()                            # 카메라 입력을 받는다. img는 Numpy 배열.
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # BGR => RGB 컬러 채널 변환.
    imgRGB = my_detector.findHands(imgRGB)            # 검출하고 시각화.
    lms = my_detector.getLandmarks(imgRGB)            # 검출된 landmark 좌표점.          

    # 검출이 성공한 경우만 출력.
    if lms:
        print(lms)

    imgBGR = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)  # RGB => BGR 역변환.
    cv2.imshow("Image", imgBGR)
    if cv2.waitKey(1) & 0xFF == ord('q'):             # 'q' 키가 눌려지면 나간다.
        break
