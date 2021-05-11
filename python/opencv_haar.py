import cv2, time

# load model
# haarcascade 모델을 불러와서 사용한다.   
detector = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# initialize video source, default 0 (webcam)
cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')


frame_count, tt = 0, 0

while cap.isOpened():
  ret, img = cap.read()
  if not ret:
    break

  frame_count += 1

  start_time = time.time()

  # prepare input
  result_img = img.copy()
  gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)

  # inference, find faces
  # 인테그랄(Integral) 이미지 사용을 위해 gray scale 이미지로 변환한다.  
  detections = detector.detectMultiScale(gray)

  # postprocessing
  # detection한 결과 top-left[왼쪽 위] 좌표와 rect의 너비[w], 높이[h] 값이 return된다.
  # 위의 값으로 bottom-right[오른쪽 아래] 좌표를 유추해낼 수 있다.  
  for (x1, y1, w, h) in detections:
    x2 = x1 + w
    y2 = y1 + h

    # draw rects
    cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)

  # inference time
  # fps(frame pre second)별 정확도 확인
  tt += time.time() - start_time
  fps = frame_count / tt
  cv2.putText(result_img, '(haar): %.2f' % (fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

  # visualize
  cv2.imshow('result', result_img)
  # esc 누를시 종료
  if cv2.waitKey(1) == 27:
    break


cap.release()
out.release()
cv2.destroyAllWindows()