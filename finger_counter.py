"""
Real-Time Finger & Hand Gesture Counting Pipeline using OpenCV.
Techniques: Running Average Background Subtraction, Morphological Filtering,
Contour Detection, Convex Hull, and Geometric Convexity Metrics.
"""

import cv2
import numpy as np
from sklearn.metrics import pairwise

# Global background model
background = None
accumulated_weight = 0.5

# Region of Interest (ROI) coordinates for webcam tracking
roi_top = 20
roi_bottom = 300
roi_right = 300
roi_left = 600


def calc_accum_avg(frame: np.ndarray, weight: float):
    """Calculates weighted running average of background frames."""
    global background
    if background is None:
        background = frame.copy().astype("float")
        return None
    cv2.accumulateWeighted(frame, background, weight)


def segment_hand(frame: np.ndarray, threshold_val: int = 25):
    """Segments hand contour from background using absolute difference and thresholding."""
    global background
    diff = cv2.absdiff(background.astype("uint8"), frame)
    _, thresh = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    # Assume largest contour in ROI is the hand
    hand_segment = max(contours, key=cv2.contourArea)
    return (thresh, hand_segment)


def count_fingers(thresholded: np.ndarray, hand_segment: np.ndarray):
    """Calculates convex hull and counts finger intersections using pairwise distance."""
    conv_hull = cv2.convexHull(hand_segment)

    # Extreme boundary points of the hand
    top = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    left = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    right = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])

    # Center of hand
    cX = int((left[0] + right[0]) / 2)
    cY = int((top[1] + bottom[1]) / 2)

    # Compute Euclidean distance to extreme points
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[left, right, top, bottom])[0]
    max_distance = distance.max()

    # Create bounding circular ROI
    radius = int(0.8 * max_distance)
    circumference = (2 * np.pi * radius)

    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    cv2.circle(circular_roi, (cX, cY), radius, 255, 10)

    # Bitwise AND to find finger slices crossing the perimeter
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    contours, _ = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        # Filter noise based on bounding dimensions and distance from wrist
        out_of_wrist = ((cY + (cY * 0.25)) > (y + h))
        limit_pts = ((circumference * 0.25) > c.shape[0])
        if out_of_wrist and limit_pts:
            count += 1

    return count


def run_pipeline():
    """Main loop for camera stream capture and processing."""
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Camera not accessible. Please verify your webcam device.")
        return

    num_frames = 0
    print("Calibrating background for 60 frames. Keep hand outside ROI...")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_copy = frame.copy()

        roi = frame[roi_top:roi_bottom, roi_right:roi_left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if num_frames < 60:
            calc_accum_avg(gray, accumulated_weight)
            if num_frames <= 59:
                cv2.putText(frame_copy, "CALIBRATING BACKGROUND...", (200, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame_copy, "CALIBRATION COMPLETE", (200, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            hand = segment_hand(gray)
            if hand is not None:
                thresholded, hand_segment = hand
                cv2.drawContours(frame_copy, [hand_segment + (roi_right, roi_top)], -1, (255, 0, 0), 1)

                fingers = count_fingers(thresholded, hand_segment)
                cv2.putText(frame_copy, f"Fingers: {fingers}", (70, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow("Thresholded Hand", thresholded)

        # Draw ROI box
        cv2.rectangle(frame_copy, (roi_left, roi_top), (roi_right, roi_bottom), (0, 255, 0), 2)
        num_frames += 1

        cv2.imshow("Hand Gesture Recognition", frame_copy)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # Esc key
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_pipeline()
