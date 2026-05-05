import streamlit as st
import cv2
from ultralytics import YOLO

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="HIM Parladka Mosque Counter",
    layout="centered"
)

# ------------------ STYLING ------------------
st.markdown("""
    <style>
    body {
        background-color: #0e2f2f;
    }
    .main {
        background-color: #0e2f2f;
        color: white;
    }
    h1, h2, h3 {
        text-align: center;
        color: #a8e6cf;
    }
    .count-box {
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        color: #ffffff;
        background: #14532d;
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.title("🕌 HIM Parladka Mosque")
st.subheader("Live People Counter")

# ------------------ MODEL ------------------
model = YOLO("yolov8n.pt")

# ------------------ CONTROLS ------------------
run = st.checkbox("Start Camera")

frame_window = st.image([])
count_placeholder = st.empty()

# ------------------ CAMERA LOGIC ------------------
if run:
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Failed to access camera")
            break

        people_count = 0  # ✅ FIX: initialize before counting

        # Run YOLO detection
        results = model(frame)

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])

                # Class 0 = person
                if cls == 0:
                    people_count += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # ------------------ DISPLAY COUNT ------------------
        count_placeholder.markdown(
            f"<div class='count-box'>People Count: {people_count}</div>",
            unsafe_allow_html=True
        )

        # ------------------ DISPLAY FRAME ------------------
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame)

    camera.release()
else:
    st.info("Click 'Start Count' to begin")
