
  
#python app_eye.py --shape-predictor shape_predictor_68_face_landmarks.dat
# -*- coding: utf-8 -*-

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import datetime
from gtts import gTTS
import tkinter as tk
from tkinter import ttk
from playsound import playsound

def playaudio(text):
    speech=gTTS(text)
    print(type(speech))
    speech.save("../output1.mp3")
    playsound("../output1.mp3")
    return

LARGE_FONT= ("Verdana", 12)
NORM_FONT = ("Helvetica", 10)
SMALL_FONT = ("Helvetica", 8)

def popupmsg(msg):
    popup = tk.Tk()
    popup.wm_title("Urgent")
    style = ttk.Style(popup)
    style.theme_use('classic')
    style.configure('Test.TLabel', background= 'aqua')
    label = ttk.Label(popup, text=msg,style= 'Test.TLabel')
    label.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text="Okay", command = popup.destroy)
    B1.pack()
    popup.mainloop()
    
    
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
	
	ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
	return ear
 # construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold 

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

#COUNTER  is the total number of successive frames that have an eye aspect ratio 
#less than EYE_AR_THRESH

#TOTAL  is the total number of blinks that have taken place while
# the script has been running

COUNTER = 0
TOTAL = 0

## initializing dlib's face detector (HOG-based) and then create
# the facial landmark predictor

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
print(type(predictor),predictor)

# grab the indexes of the facial landmarks for the left and
# right eye, respectively

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

eye_thresh = 10
before =datetime.datetime.now().minute

if not args.get("video", False):
    # Taking input from web cam
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
    #before =datetime.datetime.now().minute
    
else:
    print("[INFO] opening video file...")
    #Taking input as video file
    vs = cv2.VideoCapture(args["video"])
    time.sleep(1.0)
    #before =datetime.datetime.now().minute

# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    # loop over the face detections
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        ear = (leftEAR + rightEAR) / 2.0
        # compute the convex hull for the left and right eye, then
		# visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        # check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1
        # otherwise, the eye aspect ratio is not below the blink
		# threshold
        else:
            
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            # reset the eye frame counter
            COUNTER = 0
            
        now = datetime.datetime.now().minute
        no_of_min = now - before
        print(no_of_min, before, now)
        blinks = no_of_min * eye_thresh
        
        if(TOTAL < blinks - eye_thresh):
            playaudio("Take rest for a while as your blink count is less than average count")
            popupmsg("Take rest for a while!!!! :D")
            cv2.putText(frame, "Take rest for a while!!!! :D", (70, 150),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif(TOTAL > blinks + eye_thresh):
            playaudio("Take rest for a while as your blink count is more than average count")
            popupmsg("Take rest for a while!!!! :D")
            cv2.putText(frame, "take rest for a while!!!! :D ", (70, 150),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()

