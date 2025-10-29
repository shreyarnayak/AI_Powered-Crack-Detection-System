import streamlit as st
import pandas as pd
import cv2
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from keras.models import load_model
import os

# Load trained model
model = load_model("mini_cnn_model.keras")

# Calibration constants
PIXEL_TO_MM = 0.05
DEPTH_FACTOR = 0.8

def classify_seriousness(depth):
    if depth < 1:
        return "Minor"
    elif depth < 2.5:
        return "Moderate"
    else:
        return "Severe"

def estimate_crack_depth_from_frame(frame):
    resized = cv2.resize(frame, (227,170))
    input_img = resized / 255.0
    input_img = np.expand_dims(input_img, axis=0)

    pred = model.predict(input_img, verbose=0)[0][0]
    if pred < 0.5:
        return None

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    crack_contour = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(crack_contour)

    crack_width_px = w
    crack_width_mm = crack_width_px * PIXEL_TO_MM
    crack_depth_mm = crack_width_mm * DEPTH_FACTOR

    seriousness = classify_seriousness(crack_depth_mm)
    return crack_depth_mm, seriousness

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_count = 0
    report_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        result = estimate_crack_depth_from_frame(frame)

        if result is not None:
            depth, seriousness = result
            text = f"Depth: {depth:.2f} mm | {seriousness}"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255), 2)

            report_data.append({
                "Frame": frame_count,
                "Depth (mm)": round(depth, 2),
                "Seriousness": seriousness
            })

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print("Total frames processed:", frame_count)
    print("Report rows:", len(report_data))
    print("Output video size:", os.path.getsize(output_path), "bytes")
    return pd.DataFrame(report_data)



# ---------------- STREAMLIT DASHBOARD ---------------- #
st.set_page_config(page_title="Crack Monitoring Dashboard", layout="wide")
st.title("🛠️ Crack Monitoring Dashboard")

uploaded_video = st.file_uploader("🎥 Upload Infrastructure Video", type=["mp4", "avi", "mov"])

if uploaded_video:
    # Save uploaded video temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    input_video_path = tfile.name

    # Create unique temp output file
    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

    st.info("⏳ Processing video... please wait")
    df = process_video(input_video_path, output_path=output_video_path)

    if not df.empty:
        st.success("✅ Crack analysis complete!")

        # Debug file size
        if os.path.exists(output_video_path):
            st.write("Video saved. Size:", os.path.getsize(output_video_path), "bytes")

        # Show annotated video
        st.subheader("🎥 Annotated Video")
        with open(output_video_path, "rb") as f:
            st.video(f.read())

        # Show table
        st.subheader("📄 Crack Report")
        st.dataframe(df)

        # Plot Depth vs Frame
        st.subheader("📈 Depth vs Frame")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(df["Frame"], df["Depth (mm)"], marker="o", linestyle="-", color="red")
        ax.set_title("Crack Depth Across Frames")
        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Depth (mm)")
        ax.grid(True)
        st.pyplot(fig)

        # Summary
        st.subheader("📊 Summary")
        max_depth = df["Depth (mm)"].max()
        avg_depth = df["Depth (mm)"].mean()
        most_serious = df["Seriousness"].apply(lambda x: ["Minor","Moderate","Severe"].index(x)).max()
        most_serious_label = ["Minor","Moderate","Severe"][most_serious]

        st.metric("Max Depth (mm)", f"{max_depth:.2f}")
        st.metric("Average Depth (mm)", f"{avg_depth:.2f}")
        st.metric("Most Serious Crack", most_serious_label)

        if most_serious_label == "Severe":
            st.error("🚨 ALERT: Severe crack detected! Immediate attention required.")
    else:
        st.warning("⚠️ No cracks detected in the video.")
