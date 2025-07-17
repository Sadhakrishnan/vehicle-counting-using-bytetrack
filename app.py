import streamlit as st
import os
import uuid
from vehicle_tracking import run_vehicle_tracking

os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

st.set_page_config(page_title="Vehicle Tracker", layout="wide")
st.title("🚦 Vehicle Tracking and Counting App")

uploaded_file = st.file_uploader("📤 Upload a video", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    file_id = str(uuid.uuid4())
    input_path = os.path.join("uploads", f"{file_id}.mp4")
    output_path = os.path.join("outputs", f"tracked_{file_id}.mp4")

    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    st.video(input_path)

    if st.button("▶️ Start Tracking"):
        with st.spinner("Running detection..."):
            out_path, vehicle_counts = run_vehicle_tracking(input_path, output_path)

        st.success("✅ Done!")
        st.video(out_path)

        st.subheader("📊 Vehicle Count Summary")
        for v, c in vehicle_counts.items():
            st.write(f"- **{v.capitalize()}**: {c}")
        st.write(f"### 🧮 Total: {sum(vehicle_counts.values())}")

        with open(out_path, "rb") as f:
            st.download_button("📥 Download Processed Video", f, file_name="tracked_video.mp4")
