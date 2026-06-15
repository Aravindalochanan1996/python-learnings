import cv2
vs = cv2.VideoCapture(0)
while True:
    _,frame = vs.read()
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(1)
    print(key)
    if key == ord('a'):
        break
vs.release()
cv2.destroyAllWindows()
    
