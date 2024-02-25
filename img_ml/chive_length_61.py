import cv2
import numpy as np

def find_leaves_and_measure(image_path):
    # Load the image
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define a range for the color of the leaves and create a mask
    lower_green = np.array([25, 52, 72])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Find contours
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lengths = []
    widths = []

    for contour in contours:
        # Skip small contours that may be noise
        if cv2.contourArea(contour) < 100:
            continue

        # Fit a line to the contour
        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        slope = vy / vx
        intercept = y - (slope * x)

        # Get endpoints of the line for drawing
        start_point = (0, int(intercept))
        end_point = (image.shape[1], int(slope * image.shape[1] + intercept))
        cv2.line(image, start_point, end_point, (0, 0, 255), 2)

        # Measure the length of the contour
        length = cv2.arcLength(contour, closed=False)
        lengths.append(length)

        # Measure the width as the maximum distance between points on the contour
        width = max([np.linalg.norm(np.array(p1) - np.array(p2)) for p1 in contour for p2 in contour])
        widths.append(width)

    average_length = np.mean(lengths) if lengths else 0
    average_width = np.mean(widths) if widths else 0

    # Save the result
    output_path = 'img/processed_chives.png'
    cv2.imwrite(output_path, image)

    return average_length, average_width, output_path

# Path to the image file
image_path = 'img/chive_s1.jpeg'

# Process the image and output results
average_length, average_width, processed_image_path = find_leaves_and_measure(image_path)
average_length, average_width, processed_image_path
