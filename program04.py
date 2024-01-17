#
# 각도 추출.
#

import cv2
import numpy as np
import handTracking as ht                   # 준비해 둔 모듈을 읽어온다.

# Landmarks를 받아서 마디 벡터 사이의 각도를 반환해 주는 함수.
def getAngles(lms):
    base = lms[0][1:]             # 0번 landmark의 가로, 세로 좌표.
    lms = np.array( [ (x,y) for id, x, y in lms  ] )
    vectors = lms[1:] - np.array([base]*20)                                   # 마디와 마디를 연결해서 벡터를 만든다.
    norms = np.linalg.norm(vectors, axis=1)[:, np.newaxis]           # 축의 수가 2개 되도록 구성된 norm. 
    vectors = vectors/norms                                          # 길이가 1인 벡터로 정규화.
    cos = np.einsum( 'ij,ij->i', vectors[:-1], vectors[1:])
    angles = np.arccos(cos)*180/np.pi                                # Radian => Degree 변환.
    return angles   

my_cap = cv2.VideoCapture(0)                  # 0 = 첫번째 카메라.
my_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)     # 카메라 입력의 너비.
my_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)    # 카메라 입력의 높이.
my_detector = ht.HandDetector()               # 검출 객체 생성.
while True:
    _, img = my_cap.read()               # 카메라 입력을 받는다.
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # BGR => RGB 컬러 채널 변환.
    imgRGB = my_detector.findHands(imgRGB)            # 검출하고 시각화.
    lms = my_detector.getLandmarks(imgRGB)            # 검출된 landmark 좌표점.          

    # 검출이 성공한 경우만 출력.
    if lms:
        print(getAngles(lms))              
        
    imgBGR = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)  # RGB => BGR 역변환.
    cv2.imshow("Image", imgBGR)
    if cv2.waitKey(1) & 0xFF == ord('q'):             # 'q' 키가 눌려지면 나간다.
        break

