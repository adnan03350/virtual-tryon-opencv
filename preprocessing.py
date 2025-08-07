# preprocessing.py

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose_estimator = mp_pose.Pose(static_image_mode=True)

def detect_pose(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_estimator.process(image_rgb)

    if not results.pose_landmarks:
        print("⚠️ No pose landmarks detected.")
        return None, image

    annotated_image = image.copy()
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing.draw_landmarks(
        annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
    )

    # Get keypoints as (x, y)
    h, w, _ = image.shape
    keypoints = []
    for lm in results.pose_landmarks.landmark:
        keypoints.append((int(lm.x * w), int(lm.y * h)))

    return keypoints, annotated_image
