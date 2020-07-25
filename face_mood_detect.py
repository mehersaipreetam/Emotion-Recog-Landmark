import dlib
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

#For face rectangles:
detector = dlib.get_frontal_face_detector()

# Here, we need shape_predictor_58_face_landmarks.dat
# Provided by dlib itself
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

data = np.load('face_mood.npy')
X = data[:,1:].astype(int)
y = data[:,0]

model = KNeighborsClassifier()
model.fit(X,y)

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

        out = model.predict([expression.flatten()])


        cv2.putText(frame, str(out[0]), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)


    if ret:
        cv2.imshow('My Frame',frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
