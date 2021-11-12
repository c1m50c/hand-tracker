from dataclasses import dataclass
from typing import List, Tuple
from numpy import ndarray
import mediapipe as mp
import cv2


# See "google.github.io/mediapipe/images/mobile/hand_landmarks.png" for point reference
FINGER_TIPS = [ 4, 8, 12, 16, 20 ]
MP_DRAW = mp.solutions.drawing_utils
MP_HANDS = mp.solutions.hands


@dataclass
class TrackerColors:
    """
        ## Fields
        ```py
        point_color: Tuple[int; 3] # Color of the hand's points
        connection_color: Tuple[int; 3] # Colors of the lines connecting the hand's points
        finger_tip_up_color: Tuple[int; 3] # Color of the circle surrounding a finger tip's point when it is up
        finger_tip_down_color: Tuple[int; 3] # Color of the circle surrounding a finger tip's point when it is down
        ```
    """
    
    point_color: Tuple[int]
    connection_color: Tuple[int]
    finger_tip_up_color: Tuple[int]
    finger_tip_down_color: Tuple[int]


class HandTracker(object):
    """
        Class for processing an image and tracking the hands within it.
        
        ## Fields:
        ```py
        hand_count: int # Number of hands in frame, number of hands to track determined by `hands`
        finger_count: int # Number of Finger tips pointing up
        colors: TrackerColors # Colors for the tracker
        hands: Hands # Media Pipe Hands for tracking settings
        ```
    """
    
    hand_count: int
    finger_count: int
    colors: TrackerColors
    hands: MP_HANDS.Hands
    
    def __init__(self, colors: TrackerColors, hands: MP_HANDS.Hands = MP_HANDS.Hands()) -> None:
        """
            Class for processing an image and tracking the hands within it.
            
            ## Parameters:
            ```py
            colors: TrackerColors # Colors for the tracker
            hands: Hands = Hands() # Media Pipe Hands Class for tracking settings
            ```
        """
        
        self.hands = hands
        self.colors = colors
        self.hand_count = 0
        self.finger_count = 0
        super().__init__()
    
    def process(self, image: ndarray) -> None:
        """
            Process the `image` for hands, manipulating the image with received information.
            
            ## Parameters:
            ```py
            image: ndarray # Image to be processed
            ```
        """
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        self.hand_count = 0 # Number of Hands in Frame
        self.finger_count = 0 # Number of Fingers being help up
        if results.multi_hand_landmarks:
            self.hand_count = len(results.multi_hand_landmarks)
            height, width, _ = image.shape

            for hand_lm in results.multi_hand_landmarks:
                # Get average of hand's `MCP.y` positional values for calcualting if finger is up
                MCPS_Y: List[float] = [
                    hand_lm.landmark[5].y, hand_lm.landmark[9].y, hand_lm.landmark[13].y, hand_lm.landmark[17].y]
                MCP_Y_AVG: float = sum(MCPS_Y) / len(MCPS_Y)
                
                # For each hand visible draw the connections and landmarks
                MP_DRAW.draw_landmarks(
                    image=image,
                    landmark_list=hand_lm,
                    connections=MP_HANDS.HAND_CONNECTIONS,
                    landmark_drawing_spec = MP_DRAW.DrawingSpec(color=self.colors.point_color), # Point Spec
                    connection_drawing_spec = MP_DRAW.DrawingSpec(color=self.colors.connection_color), # Line Spec
                )
                
                # For each hand visible draw finger tips,
                # and calculate how many fingers are help up
                for idx, lm in enumerate(hand_lm.landmark):
                    x, y = int(lm.x * width), int(lm.y * height)
                    is_up_color = self.colors.finger_tip_up_color # Finger is Up
                    
                    if idx in FINGER_TIPS:
                        if MCP_Y_AVG < lm.y:
                            # Finger is down
                            is_up_color = self.colors.finger_tip_down_color
                        else:
                            self.finger_count += 1
                        
                        cv2.circle(
                            img=image,
                            center=(x, y),
                            radius=5,
                            color=is_up_color,
                            thickness=2,
                            lineType=cv2.LINE_4
                        )