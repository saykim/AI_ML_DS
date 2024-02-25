import cv2
import numpy as np

def preprocess_image(image_path):
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
    
    # 잡음 및 작은 객체 제거
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
    
    return image, contours

def draw_measurement_lines(image, rect):
    # 사각형의 중심점, 너비, 높이, 회전 각도를 가져옴
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # 사각형 그리기
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

    # 사각형의 중심점과 각 모서리점 사이에 선 그리기
    center = tuple(np.int0(rect[0]))
    for p in box:
        cv2.line(image, center, tuple(p), (255, 0, 0), 2)
    
    return image

def measure_contours(image, contours):
    measurements = []
    for cnt in contours:
        # 쳐진 부분을 포함하여 윤곽선 따라 전체 길이 측정
        length = cv2.arcLength(cnt, closed=False)
        
        # 가장 넓은 부분에서 너비 측정
        rect = cv2.minAreaRect(cnt)
        width = max(rect[1])
        
        # 측정 결과 저장
        measurements.append((length, width))
        
        # 윤곽 및 측정 길이 표시
        cv2.drawContours(image, [cnt], 0, (0, 0, 255), 2)
        
        # 측정된 길이와 너비를 이미지에 선으로 표시
        image = draw_measurement_lines(image, rect)
    
    return image, measurements

# 이미지 처리 파이프라인
image_path = 'img/chive_s1.jpeg'
processed_image, contours = preprocess_image(image_path)
result_image, measurements = measure_contours(processed_image, contours)

# 결과 이미지 저장
output_path = 'img/measured_chives_with_lines.png'
cv2.imwrite(output_path, result_image)

# 측정 결과 출력
for i, (length, width) in enumerate(measurements):
    print(f"Leaf {i}: Length = {length} pixels, Width = {width} pixels")

# 결과를 실제 치수로 변환하기 위한 스케일 팩터 (이 값은 측정된 스케일 바를 기반으로 설정해야 함)
scale_factor = 0.1  # 예시: 1 픽셀 = 0.1mm

# 측정값을 실제 치수로 변환
real_measurements = [(length * scale_factor, width * scale_factor) for length, width in measurements]

# 실제 치수 출력
for i, (real_length, real_width) in enumerate(real_measurements):
    print(f"Leaf {i}: Real Length = {real_length} mm, Real Width = {real_width} mm")

# 끝으로, 이미지에 선을 포함한 결과 이미지를 확인합니다.
print(f"Processed image saved to: {output_path}")
