import dlib
import cv2
import numpy as np
import os

face_lms = []
mood = []

exp = input('Enter your mood: ')
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

        expression = np.array([[point.x - face.left(), point.y - face.top()] for point in landmarks.parts()[17:]])

    if ret:
        cv2.imshow('My Frame',frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

    if key == ord("c"): # capture
        #cv2.imwrite(name + '.jpg',frame)
        face_lms.append(expression.flatten())
        mood.append([exp])

X = np.array(face_lms)
y = np.array(mood)

data = np.hstack([y,X])
file_name = 'face_mood.npy'

if os.path.exists(file_name):
    old = np.load(file_name)
    data = np.vstack([old, data])

np.save(file_name, data)

cap.release()
cv2.destroyAllWindows()
