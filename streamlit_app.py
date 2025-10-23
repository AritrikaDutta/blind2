import streamlit as st
import cv2
import tempfile
import os
import time
from video_stream_tracking_appmodule import process_frame, fps
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
from voice_feedback_2 import VoiceAlertManager

# --- Streamlit Page Config ---
st.set_page_config(page_title="Pedestrian Safety Assistant", layout="wide")
st.title("🚦 Pedestrian Safety Assistant")
st.markdown("Upload a video or use your live camera feed to analyze crossing safety.")

# --- Mode Selection ---
mode = st.radio("Choose input source:", ["Upload Video", "Live Camera"])

if mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(uploaded_file.read())
        temp_video.close()

        cap = cv2.VideoCapture(temp_video.name)

        if not cap.isOpened():
            st.error("❌ Could not open the uploaded video.")
        else:
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            if video_fps > 0:
                fps = int(video_fps)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out_path = tempfile.NamedTemporaryFile(delete=False, suffix="_processed.mp4").name
            out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

            frame_counter = 0
            voice_alert = VoiceAlertManager()
            with st.spinner("Processing video frames..."):
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_counter += 1
                    processed_frame = process_frame(frame)
                    # After processing, get the audio path for the current label
                    audio_path = voice_alert.get_audio_path("Move")  # or "Stop", depending on your logic
                cap.release()
                out.release()
            st.success(f"✅ Video processing completed. {frame_counter} frames processed.")
            st.video(out_path)
            # Play the last generated audio (example)
            st.audio(audio_path)
            os.unlink(temp_video.name)
            os.unlink(out_path)
    else:
        st.info("👆 Upload a video file to start analysis.")

elif mode == "Live Camera":
    camera_mode = st.radio("Camera Mode:", ["Local (OpenCV)", "Web (Browser)"])
    
    if camera_mode == "Local (OpenCV)":
        stframe = st.empty()
        run = st.checkbox("Start Camera")

        if run:
            cap = cv2.VideoCapture(0)  # 0 = default webcam

            if not cap.isOpened():
                st.error("❌ Could not access your camera.")
            else:
                while run and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    processed_frame = process_frame(frame)
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    stframe.image(rgb_frame, channels="RGB", use_container_width=True)

                    time.sleep(1 / fps)

                cap.release()
                st.success("✅ Camera analysis stopped.")
    
    else:  # Web (Browser)
        st.info("📹 Click 'START' to begin browser camera feed (works on web deployments)")
        if "last_label" not in st.session_state:
            st.session_state.last_label = None
        def video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")
            processed_frame = process_frame(img)
            label = "Move"  
            if st.session_state.last_label != label:
                audio_path = voice_alert.get_audio_path(label)
                st.session_state.last_label = label
                st.audio(audio_path)
            return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")
        webrtc_ctx = webrtc_streamer(
            key="pedestrian-safety-web",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
