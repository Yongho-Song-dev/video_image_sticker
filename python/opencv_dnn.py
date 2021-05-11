import cv2, time

# load model
model_path = 'models/opencv_face_detector_uint8.pb'
config_path = 'models/opencv_face_detector.pbtxt'
net = cv2.dnn.readNetFromTensorflow(model_path, config_path)

conf_threshold = 0.7

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
  h, w, _ = result_img.shape
  blob = cv2.dnn.blobFromImage(result_img, 1.0, (640, 360), [104, 117, 123], False, False)
  net.setInput(blob)

  # inference, find faces
  detections = net.forward()

  # postprocessing
    # (x1,y1) : 얼굴인식한 rect에서의 왼쪽 위 좌표
    # (x2,y2) : 얼굴인식한 rect에서의 오른쪽 아래 좌표
  for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    # conf_threshold = 0.7   : 0.7 확률 초과인 결과만 얼굴로 판단
    if confidence > conf_threshold:
      x1 = int(detections[0, 0, i, 3] * w)
      y1 = int(detections[0, 0, i, 4] * h)
      x2 = int(detections[0, 0, i, 5] * w)
      y2 = int(detections[0, 0, i, 6] * h)

      # draw rects
      # top-left corner (x1, y1) , bottom-right corner (x2, y2)  box 그리기
      cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 255, 255), int(round(h/150)), cv2.LINE_AA)
      # 위의 결과를 확률로 표시 
      cv2.putText(result_img, '%.2f%%' % (confidence * 100.), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

  # inference time
  tt += time.time() - start_time
  fps = frame_count / tt
  cv2.putText(result_img, 'FPS(dnn): %.2f' % (fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

  # visualize
  cv2.imshow('result', result_img)
    # esc 누를시 종료
  if cv2.waitKey(1) == 27:
    break


cap.release()
out.release()
cv2.destroyAllWindows()
