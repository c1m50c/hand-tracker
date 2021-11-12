import mediapipe as mp
import time
import cv2


if __name__ == "__main__":
    FINGER_TIPS = [ 4, 8, 12, 16, 20 ]
    
    capture = cv2.VideoCapture(0)
    mp_draw = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    previous_time = 0
    current_time = 0

    while True:
        success, image = capture.read()
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        hand_count: int = 0
        number_count: int = 0
        image.flags.writeable = True
        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
            for hand_lm in results.multi_hand_landmarks:
                # For each hand visible draw finger tips
                for idx, lm in enumerate(hand_lm.landmark):
                    height, width, _ = image.shape
                    x, y = int(lm.x * width), int(lm.y * height)
                    if idx in FINGER_TIPS:
                        cv2.circle(
                            img=image,
                            center=(x, y),
                            radius=10,
                            color=(0, 0, 255),
                            thickness=2,
                        )
                
                # For each hand visible draw the connections and landmarks
                mp_draw.draw_landmarks(
                    image=image,
                    landmark_list=hand_lm,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec = mp_draw.DrawingSpec(color=(255, 255, 255)), # Point Spec
                    connection_drawing_spec = mp_draw.DrawingSpec(color=(255, 0, 0)), # Line Spec
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
            color=(255, 255, 255),
            thickness=1,
        )
        
        # Hand Count
        cv2.putText(
            img=image,
            text=f"Hands: {hand_count}",
            org=(10, 35),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1,
            color=(255, 255, 255),
            thickness=1,
        )
        
        # Number Count
        cv2.putText(
            img=image,
            text=f"Number: {number_count}",
            org=(10, 50),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1,
            color=(255, 255, 255),
            thickness=1,
        )
        
        cv2.imshow("Hand Tracker", image)
        cv2.waitKey(1)