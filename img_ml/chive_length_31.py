import cv2
import numpy as np

def calculate_contour_length(contour):
    # Calculate the length of the contour by summing the distances between each pair of consecutive points
    return cv2.arcLength(contour, closed=False)

def process_image(image_path):
    # 이미지 로드
    image = cv2.imread(image_path)
    # 흑백으로 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Converted to grayscale.")

    # 반전 이미지
    inverted_image = cv2.bitwise_not(gray_image)
    print("Inverted colors.")

    # 엣지 검출
    edges = cv2.Canny(inverted_image, 100, 200)
    print("Detected edges.")

    # 윤곽선 찾기
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours.")

    # 윤곽선 길이 측정 및 선 그리기
    lengths = []
    for contour in contours:
        length = calculate_contour_length(contour)
        lengths.append(length)
        # Draw the contour with lines
        for i in range(len(contour) - 1):
            cv2.line(image, tuple(contour[i][0]), tuple(contour[i+1][0]), (0, 255, 0), 2)
    print("Measured contour lengths.")

    # 가장 긴 선, 중간 길이의 선, 가장 짧은 선의 평균 값 계산
    lengths.sort()
    long_avg = np.mean(lengths[-len(lengths)//3:])  # Top third longest lengths
    short_avg = np.mean(lengths[:len(lengths)//3])   # Bottom third shortest lengths
    middle_avg = np.mean(lengths[len(lengths)//3:2*len(lengths)//3])  # Middle third lengths

    return long_avg, middle_avg, short_avg

# 이미지 경로를 입력하세요
image_path = 'img/chive_s1.jpeg'
longest, middle, shortest = process_image(image_path)
print(f"Longest average: {longest} pixels")
print(f"Middle average: {middle} pixels")
print(f"Shortest average: {shortest} pixels")
