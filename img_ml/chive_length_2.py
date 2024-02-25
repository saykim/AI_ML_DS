import cv2
import numpy as np

def process_image(image_path):
    # 이미지 로드
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 부추 잎을 위한 색상 범위 정의
    lower_green = np.array([30, 100, 100])
    upper_green = np.array([85, 255, 255])

    # 색상 기반 마스크 생성
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    ret, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

    # 엣지 검출 및 윤곽 추출
    edges = cv2.Canny(binary, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # 윤곽을 기반으로 사각형 계산
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 잎의 길이와 너비 계산
        width, height = rect[1]
        if width > 0 and height > 0:
            print(f"Leaf Width: {width} pixels, Leaf Height: {height} pixels")

        # 윤곽 및 사각형 그리기
        cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

    # 결과 이미지 표시
    cv2.imshow('Processed Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 이미지 경로를 입력하세요
image_path = 'img/chive_s1.jpeg'
process_image(image_path)
