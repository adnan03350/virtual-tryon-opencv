# app.py

import streamlit as st
import os
from PIL import Image
from tryon import apply_clothes
from preprocessing import detect_pose

st.title("🧥 Virtual Try-On Demo")

# Upload person
person_file = st.file_uploader("Upload a person image", type=["jpg", "png"])
cloth_file = st.file_uploader("Upload a clothing image", type=["jpg", "png"])

if person_file:
    person_path = os.path.join("inputs", "person.jpg")
    with open(person_path, "wb") as f:
        f.write(person_file.read())
    st.image(person_path, caption="Person Image", use_container_width=True)

if cloth_file:
    cloth_path = os.path.join("inputs", "cloth.jpg")
    with open(cloth_path, "wb") as f:
        f.write(cloth_file.read())
    st.image(cloth_path, caption="Clothing Image", use_container_width=True)

# Run try-on
if person_file and cloth_file:
    if st.button("👕 Try On"):
        result_path = apply_clothes(person_path, cloth_path)
        st.image(result_path, caption="Result", use_container_width=True)

if st.checkbox("Show Pose Landmarks"):
    keypoints, annotated = detect_pose(person_path)
    if keypoints:
        st.image(result_path, caption="Result", use_container_width=True)
