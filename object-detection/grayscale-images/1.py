import cv2

img = cv2.imread('Media.jpg')

cv2.imshow('Image', img)
cv2.imwrite('Photo.jpg', img)
cv2.waitKey(5000)
cv2.destroyAllWindows()

grayScaleImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('GrayScaleImage', grayScaleImg)
cv2.imshow('originalImage', img)
cv2.imwrite('GrayScalePhoto.jpg', grayScaleImg)
cv2.waitKey(5000)
cv2.destroyAllWindows()

print(img.shape)
print(img.size)
print(img.dtype)