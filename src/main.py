import mediapipe as mp
import time
import cv2


capture = cv2.VideoCapture(0)
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()


if __name__ == "__main__":
    while True:
        success, image = capture.read()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            # For each hand visible draw the connections and landmarks
            for hand_lm in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    image=image,
                    landmark_list=hand_lm,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec = mp_draw.DrawingSpec(color=(255, 255, 255)), # Point Spec
                    connection_drawing_spec = mp_draw.DrawingSpec(color=(0, 0, 255)), # Line Spec
                )
        
        cv2.imshow("Hand Tracker", image)
        cv2.waitKey(1)