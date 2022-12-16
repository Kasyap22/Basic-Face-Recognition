import cv2
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier

face_data = []

dict = {
    0: 'Girish',
    1: 'Sai'
}

labels = []
skip = 0
idx = 0
for file in os.listdir("data"):
    data = np.load(f"./data/{file}")
    face_data.append(data)
    l = [idx for i in range(data.shape[0])]
    labels.extend(l)
    idx += 1

X = np.concatenate(face_data, axis=0)
Y = np.array(labels)

print(X.shape, Y.shape)

knn = KNeighborsClassifier()
knn.fit(X,Y)

model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ret, frame = cam.read()

    if ret == False:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = model.detectMultiScale(frame, 1.3, 5)
    
    if len(faces)==0:
        continue

    for face in faces:
        x,y,w,h = faces[-1]
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,0), 2)
        offset = 5
        face_section = gray_frame[y-offset:y+h+offset, x-offset: x+w+offset]
        face_section = cv2.resize(face_section, (100,100))

        query = face_section.reshape(1, 10000)

        pred = knn.predict(query)[0]
        name = dict[int(pred)]
        cv2.putText(frame, name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

        cv2.imshow("My video", frame)

    key_pressed = cv2.waitKey(1) & 0xFF

    if key_pressed == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
 