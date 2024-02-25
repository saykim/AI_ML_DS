{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import cv2\n",
    "import numpy as np\n",
    "\n",
    "# 이미지 로드\n",
    "image = cv2.imread('img/chive_s1.jpeg')\n",
    "\n",
    "# HSV 색상 공간으로 변환\n",
    "hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)\n",
    "\n",
    "# 부추 색상 범위 정의\n",
    "lower_green = np.array([30, 100, 100])\n",
    "upper_green = np.array([85, 255, 255])\n",
    "\n",
    "# 색상 기반 마스크 생성\n",
    "mask = cv2.inRange(hsv_image, lower_green, upper_green)\n",
    "\n",
    "# 잡음 제거\n",
    "blurred = cv2.GaussianBlur(mask, (5, 5), 0)\n",
    "\n",
    "# 엣지 검출\n",
    "edges = cv2.Canny(blurred, 50, 150)\n",
    "\n",
    "# 윤곽 찾기\n",
    "contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)\n",
    "\n",
    "# 윤곽을 기반으로 각 잎 측정\n",
    "for cnt in contours:\n",
    "    rect = cv2.minAreaRect(cnt)\n",
    "    box = cv2.boxPoints(rect)\n",
    "    box = np.int0(box)\n",
    "    cv2.drawContours(image, [box], 0, (0, 0, 255), 2)\n",
    "\n",
    "    # 여기서 rect를 사용하여 잎의 길이 및 너비 측정 가능\n",
    "\n",
    "# 결과 이미지 출력\n",
    "cv2.imshow('Detected Chives', image)\n",
    "cv2.waitKey(0)\n",
    "cv2.waitKey(0)\n",
    "cv2.destroyAllWindows()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "ds310",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.17"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
