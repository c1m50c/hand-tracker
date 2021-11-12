from dataclasses import dataclass
from typing import Tuple
import mediapipe as mp
import cv2


FINGER_TIPS = [ 4, 8, 12, 16, 20 ]
mp_draw = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


@dataclass
class TrackerColors:
    point_color: Tuple[int]
    connection_color: Tuple[int]
    finger_tip_up_color: Tuple[int]
    finger_tip_down_color: Tuple[int]


class HandTracker(object):
    hand_count: int
    finger_count: int
    colors: TrackerColors
    hands: mp_hands.Hands
    
    def __init__(self, colors: TrackerColors, hands = mp_hands.Hands()) -> None:
        self.hands = hands
        self.colors = colors
        self.hand_count = 0
        self.finger_count = 0
        super().__init__()
    
    def process(self, image) -> None:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        hand_count: int = 0 # Number of Hands in Frame
        finger_count: int = 0 # Number of Fingers being help up
        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
            height, width, _ = image.shape

            for hand_lm in results.multi_hand_landmarks:
                # For each hand visible draw the connections and landmarks
                mp_draw.draw_landmarks(
                    image=image,
                    landmark_list=hand_lm,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec = mp_draw.DrawingSpec(color=self.colors.point_color), # Point Spec
                    connection_drawing_spec = mp_draw.DrawingSpec(color=self.colors.connection_color), # Line Spec
                )
                
                # For each hand visible draw finger tips,
                # and calculate how many fingers are help up
                for idx, lm in enumerate(hand_lm.landmark):
                    x, y = int(lm.x * width), int(lm.y * height)
                    is_up_color = self.colors.finger_tip_up_color # Finger is Up
                    
                    if idx in FINGER_TIPS:
                        dip = hand_lm.landmark[idx - 1]
                        if dip.y < lm.y:
                            # Finger is down
                            is_up_color = self.colors.finger_tip_down_color
                        else:
                            finger_count += 1
                        
                        cv2.circle(
                            img=image,
                            center=(x, y),
                            radius=5,
                            color=is_up_color,
                            thickness=2,
                            lineType=cv2.LINE_4
                        )