# drowsiness_detection_app.py
import streamlit as st
import cv2
import numpy as np
import dlib
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import pygame
import threading
import time
import os
import imutils

# Set page configuration
st.set_page_config(
    page_title="Drowsiness Detection System",
    page_icon="😴",
    layout="wide"
)

# Initialize pygame mixer for alert sound
mixer.init()

# Function to calculate eye aspect ratio
def eye_aspect_ratio(eye):
    # Calculate the vertical eye landmarks distance
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    
    # Calculate the horizontal eye landmarks distance
    C = distance.euclidean(eye[0], eye[3])
    
    # Calculate the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

# Function to play alert sound
def play_alarm():
    mixer.music.load("music.wav")
    mixer.music.play(-1)  # -1 means loop indefinitely

# Function to stop alarm
def stop_alarm():
    mixer.music.stop()

# Function to convert image to dlib-compatible format
def prepare_image_for_dlib(image):
    # Convert to uint8 if not already
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    # Ensure the image is in the correct format (8-bit grayscale)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Normalize to ensure values are in 0-255 range
    if gray.max() > 255 or gray.min() < 0:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # Final conversion to uint8
    gray = gray.astype(np.uint8)
    
    return gray

# Main function for drowsiness detection
def detect_drowsiness(frame, detector, predictor, ear_threshold, consecutive_frames, awake_time_required):
    # Make a copy of the frame to avoid modifying the original
    frame_copy = frame.copy()
    
    # Convert frame to grayscale with special handling for dlib
    gray = prepare_image_for_dlib(frame_copy)
    
    # Debug information
    st.sidebar.write(f"Frame shape: {frame.shape}")
    st.sidebar.write(f"Frame dtype: {frame.dtype}")
    st.sidebar.write(f"Gray shape: {gray.shape}")
    st.sidebar.write(f"Gray dtype: {gray.dtype}")
    
    # Detect faces
    try:
        # Convert to Python list to ensure compatibility
        faces = list(detector(gray, 0))
    except RuntimeError as e:
        st.error(f"Detection error: {e}")
        return frame, "Error", 0, 0
    
    # Initialize status and count
    status = "Awake"
    drowsy_count = 0
    awake_count = 0
    
    # Get the facial landmarks for the face region
    for face in faces:
        # Determine the facial landmarks
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        
        # Extract the left and right eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        # Calculate the EAR for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        
        # Average the EAR together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        
        # Compute the convex hull for the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        
        # Draw the contours on the frame
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        
        # Check if EAR is below threshold (eyes closed)
        if ear < ear_threshold:
            drowsy_count += 1
            awake_count = 0  # Reset awake counter when eyes are closed
            
            # If the eyes were closed for a sufficient number of frames, sound the alarm
            if drowsy_count >= consecutive_frames:
                status = "DROWSY!"
                
                # Draw an alarm on the frame
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Play the alarm sound if not already playing
                if not mixer.music.get_busy():
                    play_alarm()
        else:
            # Eyes are open
            if status == "DROWSY!":
                # If previously drowsy, increment awake counter
                awake_count += 1
                
                # Display time remaining until alarm stops
                remaining_time = awake_time_required - awake_count
                if remaining_time > 0:
                    cv2.putText(frame, f"Keep eyes open: {remaining_time} frames to stop alarm", 
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # If awake for required time, stop alarm and reset
                if awake_count >= awake_time_required:
                    status = "Awake"
                    drowsy_count = 0
                    stop_alarm()
            else:
                # Normal awake state
                drowsy_count = 0
                status = "Awake"
            
        # Display the calculated EAR on the frame
        cv2.putText(frame, f"EAR: {ear:.2f}", (frame.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display status on frame
        status_color = (0, 255, 0) if status == "Awake" else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    return frame, status, drowsy_count, awake_count

# Streamlit app
def main():
    # App title and description
    st.title("Drowsiness Detection System")
    st.markdown("""
    This application monitors your eyes to detect signs of drowsiness.
    If your eyes remain closed for too long, an alert will sound to wake you up.
    The alarm will continue until your eyes stay open for 15 seconds.
    """)
    
    # Sidebar for parameters
    st.sidebar.header("Parameters")
    ear_threshold = st.sidebar.slider("EAR Threshold", 0.1, 0.4, 0.25, 0.01)
    consecutive_frames = st.sidebar.slider("Consecutive Frames for Drowsy", 10, 50, 30, 1)
    awake_time_required = st.sidebar.slider("Frames to Stay Awake", 10, 100, 45, 5)  # ~15 seconds at 3 FPS
    
    # Camera frame size
    frame_width = st.sidebar.slider("Camera Width", 300, 800, 640)
    
    # Check if the shape predictor file exists
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.isfile(predictor_path):
        st.error(f"Error: {predictor_path} not found. Please download it and place it in the same directory as this script.")
        st.markdown("""
        You can download the shape predictor file from:
        [shape_predictor_68_face_landmarks.dat](https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2)
        
        After downloading, extract the .bz2 file and place the .dat file in the same directory as this script.
        """)
        return
    
    # Initialize dlib's face detector and facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    # Get facial landmarks indices for eyes
    global lStart, lEnd, rStart, rEnd
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
    
    # Create a placeholder for the video feed
    video_placeholder = st.empty()
    
    # Create a placeholder for status
    status_placeholder = st.empty()
    
    # Start/Stop buttons in columns
    col1, col2 = st.columns(2)
    start_button = col1.button("Start Detection")
    stop_button = col2.button("Stop Detection")
    
    # Session state to track if the webcam is running
    if 'running' not in st.session_state:
        st.session_state.running = False
    
    if start_button:
        st.session_state.running = True
    
    if stop_button:
        st.session_state.running = False
        # Make sure to stop any playing alarm
        stop_alarm()
        st.rerun()
    
    # Initialize webcam
    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
            return
        
        # Status indicators
        status = "Awake"
        drowsy_count = 0
        awake_count = 0
        
        try:
            while cap.isOpened() and st.session_state.running:
                # Read a frame from the video
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    st.error("Failed to grab frame from camera. Check camera index.")
                    break
                
                # Resize the frame to make it smaller
                frame = cv2.resize(frame, (frame_width, int(frame_width * frame.shape[0] / frame.shape[1])))
                
                # Process the frame for drowsiness detection
                frame, status, drowsy_count, awake_count = detect_drowsiness(
                    frame, detector, predictor, ear_threshold, consecutive_frames, awake_time_required
                )
                
                # Display the status
                if status == "DROWSY!":
                    status_placeholder.error(f"Status: {status} - Closed eyes detected for {drowsy_count} frames")
                else:
                    if awake_count > 0 and awake_count < awake_time_required:
                        status_placeholder.warning(f"Status: Recovering - Eyes open for {awake_count}/{awake_time_required} frames")
                    else:
                        status_placeholder.success(f"Status: {status}")
                
                # Convert the frame from BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display the resulting frame
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=False)
                
                # Add a small delay
                time.sleep(0.1)
                
                # Check if stop button was pressed
                if not st.session_state.running:
                    break
        finally:
            # Release the capture when everything is done
            cap.release()
            # Make sure to stop any playing alarm
            stop_alarm()

if __name__ == "__main__":
    main()