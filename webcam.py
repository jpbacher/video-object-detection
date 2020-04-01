import os
import time
import argparse
import numpy as np
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import cv2


arg = argparse.ArgumentParser()
arg.add_argument('-m', '--mobilenet', required=True, help='path to pre-trained model')
arg.add_argument('-c', '--confidence', type=float, default=0.75, help='min confidence')
args = vars(arg.parse_args())

labels_path = os.path.sep.join([args['mobilenet'], 'mobilenet_names.txt'])
labels = open(labels_path).read().strip().split("\n")
np.random.seed(3)
colors = np.random.uniform(0, 255, size=(len(labels), 3))

prototxt = os.path.sep.join([args['mobilenet'], 'MobileNetSSD_deploy.prototxt.txt'])
caffe_model = os.path.sep.join([args['mobilenet'], 'MobileNetSSD_deploy.caffemodel'])
print('*** loading model...')
model = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)

print('*** starting video stream...')
vs = VideoStream(src=0).start()
time.sleep(2.4)
fps = FPS().start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    model.setInput(blob)
    detections = model.forward()
    for d in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, d, 2]
        if confidence > args['confidence']:
            idx = int(detections[0, 0, d, 1])
            box = detections[0, 0, d, 3:7] * np.array([w, h, w, h])
            x_start, y_start, x_end, y_end = box.astype('int')
            label = f'{labels[idx]}: {confidence: 0.3f}'
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), colors[idx], 2)
            y = y_start -15 if y_start - 15 > 15 else y_start + 15
            cv2.putText(frame, label, (x_start, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    fps.update()
fps.stop()
print(f'*** elapsed time: {fps.elapsed()}')
print(f'*** approximate FPS: {fps.fps():0.3f}')
cv2.destroyAllWindows()
vs.stop()