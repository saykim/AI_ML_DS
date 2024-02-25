import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    edges = cv2.Canny(thresholded, 50, 150)
    return edges

def detect_and_draw_lines(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        # Approximate the contour to a polyline
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Draw the polyline on the image
        cv2.drawContours(image, [approx], 0, (255, 0, 0), 2)
        lines.append(approx)
    return lines

def measure_lines(lines):
    lengths = [cv2.arcLength(line, closed=False) for line in lines]
    return np.mean(lengths) if lengths else 0

# Main processing function
def process_image(image_path):
    preprocessed = preprocess_image(image_path)
    lines = detect_and_draw_lines(preprocessed)
    average_length = measure_lines(lines)
    return average_length

# Replace 'path_to_image.jpg' with your image file path
image_path = 'img/chive_s1.jpeg'
average_length = process_image(image_path)
print(f"Average Leaf Length: {average_length} pixels")
