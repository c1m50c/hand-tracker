import mediapipe as mp
import time
import cv2


capture = cv2.VideoCapture(0)
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

previous_time = 0
current_time = 0


if __name__ == "__main__":
    while True:
        success, image = capture.read()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        hand_count: int = 0
        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
            # For each hand visible draw the connections and landmarks
            for hand_lm in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    image=image,
                    landmark_list=hand_lm,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec = mp_draw.DrawingSpec(color=(255, 255, 255)), # Point Spec
                    connection_drawing_spec = mp_draw.DrawingSpec(color=(0, 0, 255)), # Line Spec
                )
        
        current_time = time.time()
        frame_rate = 1 / (current_time - previous_time)
        previous_time = current_time
        
        # Frame Rate
        cv2.putText(
            img=image,
            text=f"FrameRate: {round(frame_rate, 1)}",
            org=(10, 20),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1,
            color=(128, 128, 128),
            thickness=1,
        )
        
        # Hand Count
        cv2.putText(
            img=image,
            text=f"Hands: {hand_count}",
            org=(10, 35),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1,
            color=(128, 128, 128),
            thickness=1,
        )
        
        cv2.imshow("Hand Tracker", image)
        cv2.waitKey(1)