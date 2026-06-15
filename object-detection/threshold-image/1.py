import cv2

img = cv2.imread('Media.jpg')

grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#dst = cv2.threshold(src, thresh, maxval,binary, type)[1]

thresh1 = cv2.threshold(grayImg, 127, 255, cv2.THRESH_BINARY)[1]

cv2.imshow('Original', img)
cv2.imshow('Thresholded', thresh1)
cv2.waitKey(0)
cv2.destroyAllWindows()