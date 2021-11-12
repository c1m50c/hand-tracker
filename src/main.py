from hand_tracker import TrackerColors, HandTracker

import mediapipe as mp
import time
import cv2


if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    previous_time = 0
    current_time = 0
    
    tracker_colors = TrackerColors(
        point_color=(255, 255, 255),
        connection_color=(0, 0, 0),
        finger_tip_up_color=(0, 255, 0),
        finger_tip_down_color=(0, 0, 255),
    )
    
    tracker = HandTracker(colors=tracker_colors)

    while True:
        success, image = capture.read()
        tracker.process(image=image)
        cv2.imshow("Hand Tracker", image)
        cv2.waitKey(1)