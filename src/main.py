from hand_tracker import TrackerColors, HandTracker
from time import time

import mediapipe as mp
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
        
        # Calculate Frame Rate
        current_time = time()
        frame_rate = 1 / (current_time - previous_time)
        previous_time = current_time
        
        # Frame Rate Text
        cv2.putText(
            img=image,
            text=f"FrameRate: {round(frame_rate, 1)}",
            org=(10, 20),
            fontFace=cv2.QT_FONT_NORMAL,
            fontScale=0.5,
            color=(255, 255, 255),
            thickness=1,
        )
        
        # Hand Count Text
        cv2.putText(
            img=image,
            text=f"Hands: {tracker.hand_count}",
            org=(10, 35),
            fontFace=cv2.QT_FONT_NORMAL,
            fontScale=0.5,
            color=(255, 255, 255),
            thickness=1,
        )
        
        # Finger Count Text
        cv2.putText(
            img=image,
            text=f"Fingers: {tracker.finger_count}",
            org=(10, 50),
            fontFace=cv2.QT_FONT_NORMAL,
            fontScale=0.5,
            color=(255, 255, 255),
            thickness=1,
        )
        
        cv2.imshow("Hand Tracker", image)
        cv2.waitKey(1)