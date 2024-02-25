import cv2
import numpy as np

def calculate_contour_length(contour):
    # 윤곽선을 따라 길이를 계산하는 함수
    return cv2.arcLength(contour, closed=True)

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
        # 곡선 근사를 사용하여 윤곽선을 더 정확하게 표현
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 쳐진 잎의 길이를 계산
        leaf_length = calculate_contour_length(approx)

        # 너비는 가장 넓은 부분에서 측정 (예: bounding box 사용)
        rect = cv2.minAreaRect(contour)
        width = max(rect[1])

        print(f"Leaf Length: {leaf_length} pixels, Leaf Width: {width} pixels")

        # 윤곽 및 사각형 그리기
        cv2.drawContours(image, [approx], 0, (0, 0, 255), 2)

    # 결과 이미지 표시
    cv2.imshow('Processed Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 이미지 경로를 입력하세요
image_path = 'img/chive_s1.jpeg'
process_image(image_path)
