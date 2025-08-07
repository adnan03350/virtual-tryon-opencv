# pose.py

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def detect_pose(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found.")
        return None, None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    keypoints = []
    annotated_image = image.copy()

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            return None, image

        h, w = image.shape[:2]
        for lm in results.pose_landmarks.landmark:
            keypoints.append((int(lm.x * w), int(lm.y * h)))

        # Optional: draw pose
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    return keypoints, annotated_image
