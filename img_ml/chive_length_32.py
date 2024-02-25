import cv2
import numpy as np

# Function to preprocess the image and find contours
def preprocess_image(image_path):
    # Load the image in grayscale
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Invert the grayscale image
    inverted_image = cv2.bitwise_not(gray_image)

    # Apply Canny Edge Detection
    edges = cv2.Canny(inverted_image, 100, 200)

    # Find contours from the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, inverted_image

# Function to draw lines on the image from bottom to top of the detected contours
def draw_lines_and_measure(image, contours):
    lengths = []

    for contour in contours:
        # Assuming vertical orientation, bottom point will have the max y-coordinate
        bottom_point = max(contour, key=lambda point: point[0][1])
        # Top point will have the min y-coordinate
        top_point = min(contour, key=lambda point: point[0][1])

        # Draw a line from bottom to top point
        cv2.line(image, tuple(bottom_point[0]), tuple(top_point[0]), (255, 0, 0), 2)

        # Measure the distance between the points
        length = np.linalg.norm(np.array(bottom_point[0]) - np.array(top_point[0]))
        lengths.append(length)
    
    return lengths, image

# Load image and process
image_path = 'img/chive_s1.jpeg'
contours, inverted_image = preprocess_image(image_path)

# Draw lines and measure lengths
lengths, image_with_lines = draw_lines_and_measure(inverted_image, contours)

# Calculate average lengths for the longest, middle, and shortest lines
lengths.sort()
longest_avg = np.mean(lengths[-len(lengths)//3:]) if lengths else 0
middle_avg = np.mean(lengths[len(lengths)//3:-len(lengths)//3]) if lengths else 0
shortest_avg = np.mean(lengths[:len(lengths)//3]) if lengths else 0

# Save the image with lines
output_path = 'img/con_measured_chives_with_lines.png'
cv2.imwrite(output_path, image_with_lines)

# Now, let's draw the bounding boxes around each contour on the original image
for contour in contours:
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(inverted_image, [box], 0, (0, 0, 255), 2)

# Save the image with the bounding boxes
output_path_boxes = 'img/chives_with_bounding_boxes.png'
cv2.imwrite(output_path_boxes, inverted_image)

longest_avg, middle_avg, shortest_avg, output_path, output_path_boxes
