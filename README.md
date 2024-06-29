import cv2
import numpy as np

def detect_lanes(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Define region of interest
    height, width = image.shape[:2]
    mask = np.zeros_like(edges)
    roi_vertices = np.array([[(0, height), (width//2, height//2), (width, height)]], dtype=np.int32)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Apply Hough Transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, rho=2, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)
    
    # Draw detected lines on a blank image
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    # Overlay the detected lines on the original image
    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    
    return result

# Main function
def main():
    # Read video input
    cap = cv2.VideoCapture('input_video.mp4')
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            # Process frame
            processed_frame = detect_lanes(frame)
            
            # Display result
            cv2.imshow('Lane Detection', processed_frame)
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(# Task
