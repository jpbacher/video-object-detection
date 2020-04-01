import time
import argparse
import numpy as np
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import cv2


arg = argparse.ArgumentParser()
arg.add_argument('-m', '--model', required=True, help='path to pre-trained model')
arg.add_argument('-c', '--confidence', type=float, default=0.75, help='min confidence')
args = vars(arg.parse_args())

