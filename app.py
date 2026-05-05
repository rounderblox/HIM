import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# ---------------- UI ----------------
st.set_page_config(page_title="Mosque Counter", layout="centered")

st.title("🕌 HIM Parladka Mosque")
st.subheader("People Counter (Phone Camera Supported)")

# ---------------- MODEL ----------------
model = YOLO("yolov8n.pt")

# ---------------- CAMERA INPUT ----------------
img_file = st.camera_input("Take a photo")

if img_file is not None:
    image = Image.open(img_file)
    img = np.array(image)

    # Convert RGB → BGR for OpenCV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Resize (fast)
    img = cv2.resize(img, (640, 480))

    # Run detection
    results = model(img, imgsz=320)

    people_count = 0

    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:
                people_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show result
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    st.markdown(
        f"<h2 style='text-align:center;color:#16a34a;'>People Count: {people_count}</h2>",
        unsafe_allow_html=True
    )

