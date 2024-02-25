import cv2
import numpy as np
import matplotlib.pyplot as plt

#트랙바 생성
def nothing(x):
    pass


cv2.namedWindow('Binary')
cv2.createTrackbar('threshold', 'Binary', 0, 255, nothing)
cv2.setTrackbarPos('threshold', 'Binary', 127)


img = cv2.imread('img/redball.jpg', cv2.IMREAD_COLOR)
cv2.imshow('Color image', img)
cv2.waitKey(0)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray image', img_gray)
cv2.waitKey(0)

# 이진화
while (True):
    low = cv2.getTrackbarPos('threshold', 'Binary')
    
    # ret, img_binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY) #기존 코드
    ret, img_binary = cv2.threshold(img_gray, low, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('Binary image', img_binary)
    
    img_result = cv2.bitwise_and(img, img, mask=img_binary)
    cv2.imshow('Result image', img_result)
    
    if cv2.waitKey(1) & 0xFF == 27: 
        break



cv2.destroyAllWindows()