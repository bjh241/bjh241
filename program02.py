#
# Hand Tracking 최소 코드.
#

# 다음 사이트 참고해 본다.
# https://google.github.io/mediapipe/solutions/hands.html

# 먼저 커맨드라인에서 다음과 같이 라이브러리 설치 필요.
# pip install mediapipe
# pip install opencv-python-headless

import cv2 
import mediapipe as mp

# 손모양을 검출해 주는 Hands 객체 생성.
my_hands = mp.solutions.hands.Hands(                   
            static_image_mode=False,          # 트래킹 병행의 의미. Default=False.
            max_num_hands=1,                  # 손의 갯수. Default=2.
            min_detection_confidence=0.5,     # 검출 최소 한계. Default=0.5.
            min_tracking_confidence=0.5)      # 트래킹 최소 한계. Default=0.5.     

my_cap = cv2.VideoCapture(0)                  # 0 = 첫번째 카메라.
my_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)     # 카메라 입력의 너비.
my_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)    # 카메라 입력의 높이.

while True:
    _, img = my_cap.read()                           # 카메라 입력을 받는다. img는 Numpy 배열.
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    # BGR => RGB 컬러 채널 변환.
    res = my_hands.process(imgRGB)                   # 검출 처리.

    if(res.multi_hand_landmarks):
        a_hand = res.multi_hand_landmarks[0]         # 첫 번째 손만 사용.
        # 검출된 손의 landmark를 그려준다. 연결선도 그려준다.
        mp.solutions.drawing_utils.draw_landmarks(imgRGB, a_hand, mp.solutions.hands.HAND_CONNECTIONS)   

        for id, lm in enumerate(a_hand.landmark):    # 0~20까지 점 출력. 
#               print(id, lm)                        # 실수 비율 출력!
            h, w, c = img.shape                      # shape속성에서 이미지의 크기 (픽셀) 추출.
            cx, cy = int(lm.x * w), int( lm.y * h)   # 실수 비율을 실제 픽셀 포지션으로 변환.
            print(id, cx, cy)                  
            if (id == 0):                             # landmark 한개를 가져와서 서클 마킹을 해본다.
                cv2.circle(imgRGB, (cx, cy), 10, (255,0,255), cv2.FILLED)   

    imgBGR = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)  # RGB => BGR 역변환.
    cv2.imshow("Image", imgBGR)
    if cv2.waitKey(1) & 0xFF == ord('q'):                 # 'q' 키가 눌려지면 나간다.
        break

