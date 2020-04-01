import os
import time
import argparse
import numpy as np
import imutils
import cv2


arg = argparse.ArgumentParser()
arg.add_argument('-i', '--input', required=True, help='path to the input videos')
arg.add_argument('-o', '--output', required=True, help='path to the output videos')
arg.add_argument('-d', '--darknet', required=True, help='path to cfg & weight files')
arg.add_argument('-c', '--confidence', type=float, default=0.75, help='min confidence')
arg.add_argument('-t', '--nmsthresh', type=float, default=0.3, help='thresh for NMS')
args = vars(arg.parse_args())

labels_path = os.path.sep.join([args['darknet'], 'coco_names.txt'])
labels = open(labels_path).read().strip().split("\n")
np.random.seed(3)
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

cfg_path = os.path.sep.join([args['darknet'], 'yolov3.cfg'])
weights_path = os.path.sep.join([args['darknet'], 'yolov3.weights'])

print('loading in YOLOv3...')
model = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

l_name = model.getLayerNames()
l_name = [l_name[out[0] - 1] for out in model.getUnconnectedOutLayers()]

vc = cv2.VideoCapture(args['input'])
writer = None
width, height = (None, None)

try:
    prop = cv2.CAP_PROP_FRAME_COUNT
    frame_count = int(vc.get(prop))
    print(f'*** total frames in video: {frame_count}')
except:
    print('could not determine number of frames in the video')

while True:
    ret, frame = vc.read()
    if not ret:
        break
    if width is None or height is None:
        height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    start = time.time()
    out_array = model.forward(l_name)
    end = time.time()
    boxes = []
    confidences = []
    label_classes = []
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    start = time.time()
    out_array = model.forward(l_name)
    end = time.time()
    boxes = []
    confidences = []
    label_classes = []
    for out in out_array:
        for detection in out:
            scores = detection[5:]
            label = np.argmax(scores)
            confidence = scores[label]
            if confidence > args['confidence']:
                box = detection[0:4] * np.array([width, height, width, height])
                x_center, y_center, w, h = box.astype("int")
                x = int(x_center - (w / 2))
                y = int(y_center - (h / 2))
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                label_classes.append(label)
    ids = cv2.dnn.NMSBoxes(boxes, confidences, args['confidence'], args['nmsthresh'])
    if len(ids) > 0:
        for i in ids.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[label_classes[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(labels[label_classes[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(args['output'], fourcc, 30, (frame.shape[1], frame.shape[0]), True)
        if frame_count > 0:
            elapsed = (end - start)
            print(f'*** single frame took {elapsed:0.3f} seconds')
            print(f'*** estimated total time to finish: {elapsed*frame_count:0.3f} seconds')

    # write the output frame to disk
    writer.write(frame)
writer.release()
vc.release()