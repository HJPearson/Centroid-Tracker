import cv2
import numpy as np


"""
This function takes in an array, which will be an array of centroids in a frame, 
then finds the two closest centroids with the minimum difference of euclidian distances.
In other words, it finds the two closest centroids. This function need only return the
x and y values for one of the two centroids, which is assigned to the variable 'point'.
"""
def findCentroid(arr):
    n = len(arr)
    diff = 10**20
    point = 0

    for i in range(n-1):
        for j in range(i+1, n):
            if abs(np.linalg.norm(arr[i] - arr[j])) < diff:
                diff = abs(np.linalg.norm(arr[i] - arr[j]))
                point = arr[i]

    return point[0].astype(int), point[1].astype(int)


"""
This main function runs our program to track the centroid
"""
def main():
    # Open the video file
    video_capture = cv2.VideoCapture('oneCCC.wmv')
    if not video_capture.isOpened():
        print("Error opening video.")

    # Get the number of frames in the video
    nFrames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, frame = video_capture.read()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame.shape[1], frame.shape[0]))

    for i in range(nFrames-1):
        ret, frame = video_capture.read()                   # Read frame
        if not ret:
            print("Video is not being read")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)     # Convert frame to grayscale
        ret, binary_frame = cv2.threshold(frame, np.mean(frame), 255, cv2.THRESH_BINARY)    # Convert to binary

        # Define the structuring element (a kernel)
        kernel_size = 4
        structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        # Perform morphological opening to eliminate small noise regions
        opened_image = cv2.morphologyEx(binary_frame, cv2.MORPH_OPEN, structuring_element)

        # Compute all centroids, then find the desired centroid by calling our custom function
        num_labels, labeled_image, stats, centroids = cv2.connectedComponentsWithStats(opened_image, connectivity=8)
        centroid_x, centroid_y = findCentroid(centroids)

        # Convert frame to color again so the crosshair will be colored
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGRA)

        # Define the length and thickness of the crosshair lines
        line_length = 3
        line_thickness = 2
        # Draw the horizontal line (left to right)
        cv2.line(frame, (centroid_x - line_length, centroid_y), (centroid_x + line_length, centroid_y), (0, 0, 255), line_thickness)
        # Draw the vertical line (top to bottom)
        cv2.line(frame, (centroid_x, centroid_y - line_length), (centroid_x, centroid_y + line_length), (0, 0, 255), line_thickness)

        # Write the video to our output file
        output.write(frame)

        # Show frame with crosshair drawn on centroid
        cv2.imshow('Centroid Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    output.release()
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
