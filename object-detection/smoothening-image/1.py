import cv2

img = cv2.imread('media.jpg')

#dst = cv2.gaussianBlur(src, (kernel), borderType)

img1 = cv2.GaussianBlur(img, (41, 41), 0)
img2 = cv2.GaussianBlur(img, (21, 21), 0)

cv2.imshow('Original', img)
cv2.imshow('Smoothened 1', img1)
cv2.imshow('Smoothened 2', img2)