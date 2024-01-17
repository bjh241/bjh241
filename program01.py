#
# OpenCV 라이브러리 WebCam 사용 방법.
#

# 먼저 커맨드라인에서 다음과 같이 라이브러리 설치 필요.
# pip install opencv-python-headless

import cv2 

my_cap = cv2.VideoCapture(0)                  # 0 = 첫번째 카메라.
my_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)     # 카메라 입력의 너비.
my_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)    # 카메라 입력의 높이.

while True:
    _, img = my_cap.read()                           # 카메라 입력을 받는다. img는 Numpy 배열.

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):            # 'q' 키가 눌려지면 나간다. 1ms 기다림.
        break


