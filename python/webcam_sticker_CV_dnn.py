import numpy as np
import cv2
import dlib

from new_addsticker_test import img2sticker

#detector_hog = dlib.get_frontal_face_detector()
#landmark_predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')

# opencv + dnn caffe
model_path = 'models/res10_300x300_ssd_iter_140000.caffemodel'
config_path = 'models/deploy.prototxt'
net = cv2.dnn.readNetFromCaffe(config_path, model_path)
#  opencv + dnn  -no caffe
#model_path = 'models/opencv_face_detector_uint8.pb'
#config_path = 'models/opencv_face_detector.pbtxt'
#net = cv2.dnn.readNetFromTensorflow(model_path, config_path)

# opencv haarcase model
#detector = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

def main():
    cv2.namedWindow('show', 0)
    cv2.resizeWindow('show', 640, 360)
    # 컴퓨터와 연결된 웹캠 사용시 아래 
    vc = cv2.VideoCapture(0) 
    # 스마트폰 어플리케이션을 웹캠과 같이 rtmp로 스트리밍해서 사용
    #vc = cv2.VideoCapture('rtmp://rtmp.streamaxia.com/streamaxia/5467136366cf811e')
    img_sticker = cv2.imread('./images/king.png')
    #sticker_img = img_sticker.copy()
    vlen = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    print (vlen) # 웹캠은 video length 가 0 입니다.

    # 정해진 길이가 없기 때문에 while 을 주로 사용합니다.
    # for i in range(vlen):
    while vc.isOpened():
        ret, img = vc.read()
        if ret == False:
            break
        start = cv2.getTickCount()
        img = cv2.flip(img, 1)  # 보통 웹캠은 좌우 반전


        # prepare input
        result_img = img.copy()
        h, w, _ = result_img.shape
        print("webcam : ", result_img.shape)
        blob = cv2.dnn.blobFromImage(result_img, 1.0, (360, 300), [104, 117, 123], False, False)
        net.setInput(blob)

        # inference, find faces
        detections = net.forward()  

        # 스티커 메소드를 사용
        #img_result = img2sticker(result_img, img_sticker.copy(), detections)   
        
        x1 = y1 = x2 = y2 = 0
        confidence = 0
        conf_threshold = 0.8
        box_w, box_h =0,0
        print("h,w,_ : ", h,w)
        # postprocessing
        for i in range(detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                box_w =  x2 - x1
                box_h =  y2 - y1
                cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 255, 255), int(round(h/150)), cv2.LINE_AA)
                cv2.putText(result_img, '%.2f%%' % (confidence * 100.), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                #cv2.putText(result_img, f'w : {box_w}  h : {box_h}'   , (x2, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                break
         # sticker
        #print("confidence : ", confidence)
        if confidence > conf_threshold:
            #print(f"box_w {box_w}  , box_h : {box_h}")
            #print(f"img_sticker :  {img_sticker.shape}")
            
            #print("detections.shape : ", detections.shape[2])
  
            try:
                sticker_img = img_sticker.copy()
                refined_x = x1+box_w
                refined_y = y1-box_h
                if refined_y < 0 :
                    box_h = y1
                
                if refined_x > w :
                    box_w = w-x1

                sticker_img = cv2.resize(sticker_img, dsize=(box_w,box_h))
                sticker_img_alpha = 1.0
                background_alpha = 1.0 
                
                result_img[y1-box_h:y1, x1:x1+box_w] = \
                cv2.addWeighted( sticker_img[:, :, :3], sticker_img_alpha  , result_img[y1-box_h:y1, x1:x1+box_w] , background_alpha ,0)
            except: 
                
                print("\n======================\n image blending except \n======================\n")
                print("sticker_img : ", sticker_img.shape )
                print("sticker_area : ", result_img[y1:y1-box_h, x1:x1+box_w].shape )
                                                  
        time = (cv2.getTickCount() - start) / cv2.getTickFrequency() * 1000
        print ('[INFO] time: %.2fms'%time)
        
        #cv2.imshow('show', img_result)
        cv2.imshow('show', result_img)
        key = cv2.waitKey(1)
        if key == 27:
            break


if __name__ == '__main__':
    main()
