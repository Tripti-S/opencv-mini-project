import cv2 as cv
import numpy as np

#read maple leaf
img= cv.imread('photos\maple.png')
sunset=cv.imread('photos\sunset.png')
# Resize
resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
resize = cv.resize(sunset, (500,500), interpolation=cv.INTER_CUBIC)
cv.imshow('bleh', resize)
cv.imshow('Resized', resized)

# Converting to grayscale
gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
grey=cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
blank = np.zeros(gray.shape, dtype='uint8')


adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 17, 1)
cv.imshow('Adaptive Thresholding', adaptive_thresh)

contours, hierarchy = cv.findContours(image=adaptive_thresh, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
cv.drawContours(image=blank, contours=contours, contourIdx=0, color=(255,165,0), thickness=3, lineType=cv.LINE_AA)
cv.imshow('contour', blank)

x=cv.fillPoly(blank, contours, [255,255,255])
cv.imshow('contoured',x)

dst=cv.bitwise_and(grey,x)
cv.imshow('mix', dst)

x1 = cv.cvtColor(x, cv.COLOR_GRAY2BGR)
dst1=cv.bitwise_and(resize,x1)
cv.imshow('mix', dst1)

# col= cv.cvtColor(dst, cv.COLOR_GRAY2RGB)
# cv.imshow('final', col)
cv.waitKey(0)