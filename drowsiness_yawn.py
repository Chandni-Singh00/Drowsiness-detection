#python drowniness_yawn.py --webcam webcam_index
import tkinter as tk
from tkinter import Message, Text
import tkinter.ttk as ttk
import tkinter.font as font
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import pygame #For playing sound
import os
import espeakng
from HeadPose import getHeadTiltAndCoords


window = tk.Tk()
window.title("Drowsiness_Detector")
window.configure(background='white')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
message = tk.Label(
    window, text="Drowsiness-Detection-System",
    bg="green", fg="white", width=50,
    height=3, font=('times', 30, 'bold'))
 
message.place(x=100, y=20)


#Initialize Pygame and load music
pygame.mixer.init()
pygame.mixer.music.load('audio/alert.wav')

image_points = np.array([
    (359, 391),     # Nose tip 34
    (399, 561),     # Chin 9
    (337, 297),     # Left eye left corner 37
    (513, 301),     # Right eye right corne 46
    (345, 465),     # Left Mouth corner 49
    (453, 469)      # Right mouth corner 55
], dtype="double")


# def alarm(msg):
#     global alarm_status
#     global alarm_status2
#     global saying

#     while alarm_status:
#         print('call')
#         s = 'espeak "'+msg+'"'
#         os.system(s)

#     if alarm_status2:  
#         print('call')
#         saying = True
#         s = 'espeak "' + msg + '"'
#         os.system(s)
#         saying = False

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance



def drowsiness_detector():
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--webcam", type=int, default=0,
                    help="index of webcam on system")
    args = vars(ap.parse_args())
    
    EYE_AR_THRESH = 0.3
    EYE_AR_CONSEC_FRAMES = 10
    YAWN_THRESH = 20
    # alarm_status = False
    # alarm_status2 = False
    # saying = False
    COUNTER = 0
    print("-> Loading the predictor and detector...")
    #detector = dlib.get_frontal_face_detector()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    #Faster but less accurate
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    
    
    print("-> Starting Video Stream")
    vs = VideoStream(src=args["webcam"]).start()
    #vs= VideoStream(usePiCamera=True).start()       //For Raspberry Pi
    time.sleep(1.0)
    
    frame_width = 700
    frame_height = 700


    while True:

        frame = vs.read()
        frame = imutils.resize(frame, width=700)#450
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape

    #rects = detector(gray, 0)
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

    #for rect in rects:
        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
        
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            eye = final_ear(shape)
            ear = eye[0]
            leftEye = eye [1]
            rightEye = eye[2]

            distance = lip_distance(shape)

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            lip = shape[48:60]
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1

                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    pygame.mixer.music.play(-1)
                # if alarm_status == False:
                #     alarm_status = True
                #     t = Thread(target=alarm, args=('wake up sir',))
                #     t.deamon = True
                #     t.start()

                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    

            else:
                pygame.mixer.music.stop()
                COUNTER = 0
        
            # alarm_status = False

            if (distance > YAWN_THRESH):
                pygame.mixer.music.play(-1)
                cv2.putText(frame, "Yawn Alert", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # if alarm_status2 == False and saying == False:
                #     alarm_status2 = True
                #     t = Thread(target=alarm, args=('take some fresh air sir',))
                #     t.deamon = True
                #     t.start()
            else:
               pygame.mixer.music.stop()
            # alarm_status2 = False

            # if((COUNTER >= EYE_AR_CONSEC_FRAMES) and (distance > YAWN_THRESH)):
            #     pygame.mixer.music.play(-1)
            # else:
            #     pygame.mixer.music.stop()
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw each of them
            for (i, (x, y)) in enumerate(shape):
                if i == 33:
                # something to our key landmarks
                # save to our new key point list
                # i.e. keypoints = [(i,(x,y))]
                    image_points[0] = np.array([x, y], dtype='double')
                # write on frame in Green
                # cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                # cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 8:
                # something to our key landmarks
                # save to our new key point list
                # i.e. keypoints = [(i,(x,y))]
                    image_points[1] = np.array([x, y], dtype='double')
                # write on frame in Green
                # cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                # cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 36:
                # something to our key landmarks
                # save to our new key point list
                # i.e. keypoints = [(i,(x,y))]
                    image_points[2] = np.array([x, y], dtype='double')
                # write on frame in Green
                # cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                # cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 45:
                # something to our key landmarks
                # save to our new key point list
                # i.e. keypoints = [(i,(x,y))]
                    image_points[3] = np.array([x, y], dtype='double')
                # write on frame in Green
                # cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                # cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 48:
                # something to our key landmarks
                # save to our new key point list
                # i.e. keypoints = [(i,(x,y))]
                    image_points[4] = np.array([x, y], dtype='double')
                # write on frame in Green
                # cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                # cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                elif i == 54:
                # something to our key landmarks
                # save to our new key point list
                # i.e. keypoints = [(i,(x,y))]
                  image_points[5] = np.array([x, y], dtype='double')
                # write on frame in Green
                # cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                # cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            # else:
                # everything to all other landmarks
                # write on frame in Red
                # cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                # cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        #Draw the determinant image points onto the person's face
        # for p in image_points:
        #     cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

            (head_tilt_degree, start_point, end_point, 
            end_point_alt) = getHeadTiltAndCoords(size, image_points, frame_height)

        # cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
        # cv2.line(frame, start_point, end_point_alt, (0, 0, 255), 2)

            if head_tilt_degree:
                cv2.putText(frame, 'Head Tilt Degree: ' + str(head_tilt_degree[0]), (450, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            

            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        


        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()


trackImg = tk.Button(window, text="Testing",
                     command=drowsiness_detector, fg="white", bg="green",
                     width=20, height=3, activebackground="Red",
                     font=('times', 15, ' bold '))
trackImg.place(x=300, y=300)
quitWindow = tk.Button(window, text="Quit",
                       command=window.destroy, fg="white", bg="green",
                       width=20, height=3, activebackground="Red",
                       font=('times', 15, ' bold '))
quitWindow.place(x=800, y=300)
 
 
window.mainloop()        