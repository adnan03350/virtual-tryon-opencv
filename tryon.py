import cv2
import numpy as np
import os
import uuid
from scipy.interpolate import Rbf
from utils import remove_background
from pose import detect_pose


def tps_warp(source_img, src_pts, dst_pts, output_size):
    """Applies TPS warping from src_pts to dst_pts."""
    h, w = output_size
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    grid_x_flat = grid_x.flatten()
    grid_y_flat = grid_y.flatten()

    rbf_x = Rbf(dst_pts[:, 0], dst_pts[:, 1], src_pts[:, 0], function='thin_plate')
    rbf_y = Rbf(dst_pts[:, 0], dst_pts[:, 1], src_pts[:, 1], function='thin_plate')

    map_x = rbf_x(grid_x_flat, grid_y_flat).reshape((h, w)).astype(np.float32)
    map_y = rbf_y(grid_x_flat, grid_y_flat).reshape((h, w)).astype(np.float32)

    warped = cv2.remap(source_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return warped


def apply_clothes(person_path, cloth_path):
    person_img = cv2.imread(person_path)
    cloth_img = cv2.imread(cloth_path, cv2.IMREAD_UNCHANGED)

    if cloth_img is None:
        raise ValueError("❌ Cloth image not loaded.")
    if cloth_img.shape[2] == 3:
        cloth_img = cv2.cvtColor(cloth_img, cv2.COLOR_BGR2BGRA)

    person_nobg = remove_background(person_img)
    keypoints, _ = detect_pose(person_path)

    if keypoints is None or len(keypoints) < 25:
        raise ValueError("❌ Not enough keypoints detected.")

    # Keypoints
    left_shoulder = keypoints[11]
    right_shoulder = keypoints[12]
    left_hip = keypoints[23]
    right_hip = keypoints[24]

    mid_shoulder = [(left_shoulder[0] + right_shoulder[0]) // 2,
                    (left_shoulder[1] + right_shoulder[1]) // 2]
    mid_hip = [(left_hip[0] + right_hip[0]) // 2,
               (left_hip[1] + right_hip[1]) // 2]

    dst_pts = np.array([
        left_shoulder,                # Top-left
        right_shoulder,              # Top-right
        left_hip,                    # Bottom-left
        right_hip,                   # Bottom-right
        mid_shoulder,                # Mid-top
        mid_hip                      # Mid-bottom
    ], dtype=np.float32)

    h, w = cloth_img.shape[:2]
    src_pts = np.array([
        [0, 0],
        [w - 1, 0],
        [0, h - 1],
        [w - 1, h - 1],
        [w // 2, 0],
        [w // 2, h - 1]
    ], dtype=np.float32)

    warped_cloth = tps_warp(cloth_img, src_pts, dst_pts, (person_img.shape[0], person_img.shape[1]))

    # Blend using alpha
    if warped_cloth.shape[2] == 4:
        alpha = warped_cloth[:, :, 3] / 255.0
        for c in range(3):
            person_nobg[:, :, c] = (
                alpha * warped_cloth[:, :, c] +
                (1 - alpha) * person_nobg[:, :, c]
            ).astype(np.uint8)

    # Save result
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", f"{uuid.uuid4().hex}.jpg")
    cv2.imwrite(output_path, person_nobg)
    print(f"✅ Saved result to {output_path}")
    return output_path
