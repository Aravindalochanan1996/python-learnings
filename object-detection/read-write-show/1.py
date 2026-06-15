import cv2

img = cv2.imread('Media.jpg')

cv2.imshow('Image', img)
cv2.imwrite('Photo.jpg', img)
cv2.waitKey(5000)
cv2.destroyAllWindows()