#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 21:57:42 2020

@author: meher
"""

import dlib
import cv2

#For face rectangles:
detector = dlib.get_frontal_face_detector()

# Here, we need shape_predictor_58_face_landmarks.dat
# Provided by dlib itself
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    #To decrease load on RAM
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    for face in faces:
        #landmarks where? on each face in the frame
        landmarks = predictor(gray, face)
        #print(landmarks.parts())
        nose = landmarks.parts()[28]
        #print('Nose')
        #print(nose.x,nose.y)
        #cv2.circle(frame,(nose.x,nose.y),2,(255,0,0),3)

        ### For every point ###
#        for i in landmarks.parts():
#            cv2.circle(frame,(i.x,i.y),2,(255,0,0),3)
        ###                 ###
        
    #print('Face')
    #print(faces)

    if ret:
        cv2.imshow('My Frame',frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
