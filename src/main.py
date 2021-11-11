import mediapipe as mp
import time
import cv2


capture = cv2.VideoCapture(0)


if __name__ == "__main__":
    while True:
        success, image = capture.read()
        cv2.imshow("VideoCapture", image)
        cv2.waitKey(1)