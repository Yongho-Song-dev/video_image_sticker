import numpy as np
import cv2
import dlib

from newaddsticker import img2sticker

detector_hog = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

def main():
    cv2.namedWindow('show', 0)
    cv2.resizeWindow('show', 640, 360)
    # 컴퓨터와 연결된 웹캠 사용시 아래 
    vc = cv2.VideoCapture(0) 
    # 스마트폰 어플리케이션을 웹캠과 같이 rtmp로 스트리밍해서 사용
    #vc = cv2.VideoCapture('rtmp://rtmp.streamaxia.com/streamaxia/5467136366cf811e')
    img_sticker = cv2.imread('./images/king.png')

    vlen = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    print (vlen) # 웹캠은 video length 가 0 입니다.

    # 정해진 길이가 없기 때문에 while 을 주로 사용합니다.
    # for i in range(vlen):
    while True:
        ret, img = vc.read()
        if ret == False:
            break
        start = cv2.getTickCount()
        img = cv2.flip(img, 1)  # 보통 웹캠은 좌우 반전

        # 스티커 메소드를 사용
        img_result = img2sticker(img, img_sticker.copy(), detector_hog, landmark_predictor)   

        time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
        print ('[INFO] time: %.2fms'%time)
        
        cv2.imshow('show', img_result)
        key = cv2.waitKey(1)
        if key == 27:
            break


if __name__ == '__main__':
    main()
