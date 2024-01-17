#
# 가위 바위 보 손모양의 각도를 추출해서 저장해 준다.
#

# 먼저 커맨드라인에서 다음과 같이 라이브러리 설치 필요.
# pip install pandas

import cv2
import numpy as np
import pandas as pd
import time
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

# 초기화 준비.
recording = False                             # 아직은 기록중이 아님.
t_recording = 10                              # 시작 후 10초간 기록.
idx = 0                                       # 가위 = 0, 바위 = 1, 보 = 2.
data_set = []
labels = {0:'가위', 1:'바위', 2:'보'}

my_cap = cv2.VideoCapture(0)                  # 0 = 첫번째 카메라.
my_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)     # 카메라 입력의 너비.
my_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)    # 카메라 입력의 높이.
my_detector = ht.HandDetector()               # 검출 객체 생성.
while True:
    _, img = my_cap.read()               # 카메라 입력을 받는다.
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # BGR => RGB 컬러 채널 변환.
    imgRGB = my_detector.findHands(imgRGB)            # 검출하고 시각화.
    lms = my_detector.getLandmarks(imgRGB)            # 검출된 landmark 좌표점.          
                    
    imgBGR = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)  # RGB => BGR 역변환.
    cv2.imshow("Image", imgBGR)
    if cv2.waitKey(1) & 0xFF == ord('q'):             # 'q' 키가 눌려지면 나간다.
        break

    if (not recording) and (cv2.waitKey(1) & 0xFF == 32): # 'space' 키가 눌려지면 스타트.
        recording = True
        t_start = time.time()                           # 시작 시간.
        print('-'*30)
        print(f'"{labels[idx]}" 기록 시작.')
        print(f'남은 시간 = 10초.')

    if recording:
        t_now = time.time()                             # 현재 시간.
        if (t_now - t_start ) > 1:                      # 매 1초 경과했으면.
            t_recording -= 1                            # 1초 감소.
            print(f'남은 시간 = {t_recording}초.')
            t_start = t_now

            data_set.append(list(getAngles(lms)) + [idx])

            if t_recording == 0:
                print(f'"{labels[idx]}" 기록 끝.')
                print('-'*30)
                idx += 1
                recording = False
                t_recording = 10

                if idx == 3:                            # 가위, 바위, 보 모두 완료!
                    break

# 최종적으로 수집된 데이터 파일로 출력.
df = pd.DataFrame(data_set)
df.to_csv('data_train.csv', index=False)

