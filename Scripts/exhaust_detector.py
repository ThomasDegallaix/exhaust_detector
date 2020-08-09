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


T_BLINK_REF = time.time()
T_EYELID_REF = time.time()
T_MOBILE_GAZE_REF = time.time()
T_GAZE_REF = time.time()
STATE = "OK"
BLINK_FREQ_TRESHOLD = 0.70
GAZE_COLOR_THRESHOLD = 140
eyelid_counter, ear_mean = (0, 0.0)
TOTAL_GAZEOUT = 0

def blink_frequency():
    global T_BLINK_REF
    global TOTAL
    global BLINK_FREQUENCY_THRESHOLD
    global STATE
    T_BLINK_ACTUAL = time.time()
    blink_frequency = float(TOTAL/(T_BLINK_ACTUAL - T_BLINK_REF))
    if((T_BLINK_ACTUAL - T_BLINK_REF) > 10 ):
        T_BLINK_REF = T_BLINK_ACTUAL
        if (blink_frequency >= BLINK_FREQ_TRESHOLD):
            STATE = "TIRED : blink frequency"
        elif (blink_frequency < BLINK_FREQ_TRESHOLD and STATE == "TIRED : blink frequency"):
            STATE = "OK"
        TOTAL = 0
    cv2.putText(frame, "BF: {0:.2f}".format(blink_frequency), (150, 30),
    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def eyelid_check(ear, eyelid_counter, ear_mean):
    global T_EYELID_REF
    global STATE
    T_EYELID_ACTUAL = time.time()
    eyelid_counter+=1
    ear_mean = ear/eyelid_counter
    if ((T_EYELID_ACTUAL - T_EYELID_REF) > 5):
            T_EYELID_REF = T_EYELID_ACTUAL
            if (ear_mean <= 0.25):
                    STATE = "TIRED : eyelid check"
            elif (ear_mean > 0.25 and STATE == "TIRED : eyelid check"):
                    STATE = "OK"
            ear_mean = 0.0
            eyelid_counter = 0
    return (eyelid_counter, ear_mean)

def mobile_gaze(p2Color,p1Color):
    global T_MOBILE_GAZE_REF
    global STATE
    global TOTAL_GAZEOUT
    T_MOBILE_GAZE_ACTUAL = time.time()
    gazeout_frequency = TOTAL_GAZEOUT/(T_MOBILE_GAZE_ACTUAL - T_MOBILE_GAZE_REF)
    if(p1Color >= GAZE_COLOR_THRESHOLD or p2Color >= GAZE_COLOR_THRESHOLD):
            TOTAL_GAZEOUT+=1
    if ((T_MOBILE_GAZE_ACTUAL - T_MOBILE_GAZE_REF) > 10):
            T_MOBILE_GAZE_REF = T_MOBILE_GAZE_ACTUAL
            if (gazeout_frequency >= 5):
                    STATE = "TIRED : mobile gaze"
            elif (gazeout_frequency < 5 and STATE == "TIRED : mobile gaze"):
                    STATE = "OK"
            TOTAL_GAZEOUT=0

def gaze_loss(rects):
    global T_GAZE_REF
    T_GAZE_ACTUAL = time.time()
    if (str(rects) == "rectangles[]" and (T_GAZE_ACTUAL - T_GAZE_REF) > 2) :
            print("ALERT : GAZE LOSS")
            T_GAZE_REF = T_GAZE_ACTUAL
    elif ((T_GAZE_ACTUAL - T_GAZE_REF) > 5):
            T_GAZE_REF = T_GAZE_ACTUAL

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
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
# vs = FileVideoStream(args["video"]).start()
# fileStream = True
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)

# loop over frames from the video stream
while True:
    # if this is a file video stream, then we need to check if
    # there any more frames left in the buffer to process
    if fileStream and not vs.more():
    	break

    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # Check if the gaze can be found
    gaze_loss(rects)

    # loop over the face detections
    for rect in rects:
    	# determine the facial landmarks for the face region, then
    	# convert the facial landmark (x, y)-coordinates to a NumPy
    	# array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

    	# extract the left and right eye coordinates, then use the
    	# coordinates to compute the eye aspect ratio for both eyes
    	#print("oeil gauche start : "+str(lStart)+", oeil gauche end : "+str(lEnd))
        leftEye = shape[lStart:lEnd]
    	#print("leftEye data : "+str(leftEye))
    	#print("oeil droit start : "+str(rStart)+", oeil droit end : "+str(rEnd))
        rightEye = shape[rStart:rEnd]
    	#print("rightEye data : "+str(rightEye))
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # get 2 complementary points at 40 and 60% in the eye frame
        pointLeft = shape[36].copy()
        pointRight = shape[39].copy()
        pointUpperLeft = shape[37].copy()
        pointLowerLeft = shape[41].copy()
        pointUpperRight = shape[38].copy()
        pointLowerRight = shape[40].copy()
        xDistance = dist.euclidean(pointRight,pointLeft)
        yLDistance = dist.euclidean(pointLowerLeft,pointUpperLeft)
        yRDistance = dist.euclidean(pointLowerRight,pointUpperRight)
        pointR1 = [pointLeft[0],pointUpperLeft[1]]
        pointR1[0]+=int(xDistance*0.4)
        pointR1[1]+=int(yLDistance/2)
        pointR2 = [pointRight[0],pointUpperRight[1]]
        pointR2[0]-=int(xDistance*0.4)
        pointR2[1]+=int(yRDistance/2)

        cv2.circle(frame, (pointR1[0],pointR1[1]), 1, (0,0,255))
        cv2.circle(frame, (pointR2[0],pointR2[1]), 1, (0,0,255))

        try :
            L1color = gray[pointR1[1], pointR1[0]]
            L2color = gray[pointR2[1], pointR2[0]]
            print("Lcoord x = " +str(pointR1[0])+ ", y = " +str(pointR1[1]))
            print("Rcoord x = " +str(pointR2[0])+ ", y = " +str(pointR2[1]))
            print("L1color : "+str(L1color))
            print("L2color : "+str(L2color))
        except IndexError :
            print("IndexError")


    	# average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

    	#Calcul and print blink frequency, tell if we are tired if blink frequency too high
        blink_frequency()

    	#Check EAR mean during a certain period of time, tell if we are tired if EAR too low
        eyelid_check(ear,eyelid_counter, ear_mean)

        #Check if the gaze is mobile
        mobile_gaze(L1color, L2color)

        cv2.putText(frame, "State: {}".format(STATE), (10, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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
    		# if the eyes were closed for a sufficient number of
    		# then increment the total number of blinks
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1

    		# reset the eye frame counter
            COUNTER = 0

    	# draw the total number of blinks on the frame along with
    	# the computed eye aspect ratio for the frame
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
    		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
    	break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
