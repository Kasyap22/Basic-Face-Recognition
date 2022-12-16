import cv2

model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
    ret, frame = cam.read()

    if ret == False:
        continue

    faces = model.detectMultiScale(frame, 1.3, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,0), 2)
    
    cv2.imshow("My video", frame)

    key_pressed = cv2.waitKey(1) & 0xFF

    if key_pressed == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
 