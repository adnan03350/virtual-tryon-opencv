# 👕 Virtual Try-On using OpenCV + MediaPipe

This is a simple Virtual Try-On web application built using **OpenCV**, **MediaPipe**, and **Streamlit**. It allows users to upload a person's image and a clothing image (top/upper garment), and virtually applies the garment on the person based on their body pose.

---

## ✨ Features

- 🔍 Background removal using `rembg`
- 🎯 Pose detection using MediaPipe (33 keypoints)
- 👕 Cloth overlay with TPS (Thin Plate Spline) warping
- 📷 Streamlit web interface to upload and preview results
- ✅ Lightweight: no need for heavy models like CP-VTON

---

## 🖥️ Requirements

Tested on:
- Python 3.11
- No GPU required ✅
- Works on Windows / Mac (tested on MacBook 2019)

Install dependencies:

```bash
pip install -r requirements.txt
