import imutils
import cv2

img = cv2.imread('Media.jpg')
print(img.shape)
print(img.size)
resizeImg = imutils.resize(img, width=500)
cv2.imshow('Resized Image', resizeImg)
cv2.imwrite('ResizedPhoto.jpg', resizeImg)
cv2.waitKey(5000)
cv2.destroyAllWindows()