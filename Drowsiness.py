import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
import time
import os
from pygame import mixer
from scipy.spatial import distance

st.set_page_config(
    page_title="Advanced Drowsiness Detection",
    page_icon="😴",
    layout="wide"
)

np_version = np.__version__
if np_version.startswith('2.'):
    st.warning(f"Warning: Using NumPy {np_version}. Consider downgrading to 1.23.5: `pip install numpy==1.23.5`")

mixer.init()

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MOUTH_INDICES = [61, 291, 0, 17, 269, 405]

def euclidean_distance(point1, point2):
    return distance.euclidean(point1, point2)

def eye_aspect_ratio(eye_landmarks):
    A = euclidean_distance(eye_landmarks[1], eye_landmarks[5])
    B = euclidean_distance(eye_landmarks[2], eye_landmarks[4])
    C = euclidean_distance(eye_landmarks[0], eye_landmarks[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth_landmarks):
    A = euclidean_distance(mouth_landmarks[1], mouth_landmarks[5])
    B = euclidean_distance(mouth_landmarks[2], mouth_landmarks[4])
    C = euclidean_distance(mouth_landmarks[0], mouth_landmarks[3])
    return (A + B) / (2.0 * C)

def head_pose_estimation(landmarks, frame_height):
    nose = landmarks[1]  # Nose tip
    chin = landmarks[152]  # Chin
    face_height = euclidean_distance(nose, chin)
    nose_y = nose[1] / frame_height
    is_nodding = nose_y > 0.55 + (face_height / frame_height) * 0.1  # Dynamic threshold
    return is_nodding

def play_alarm():
    try:
        mixer.music.load("music.wav")
        mixer.music.play(-1)
    except Exception as e:
        st.error(f"Error playing alarm: {e}")

def stop_alarm():
    try:
        mixer.music.stop()
    except:
        pass

def prepare_image(frame):
    if frame is None or frame.size == 0:
        return None
    frame = frame.astype(np.uint8)
    if frame.max() > 255 or frame.min() < 0:
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return frame

def draw_timer(frame, seconds_remaining, frame_width, frame_height):
    overlay = frame.copy()
    timer_width, timer_height = 300, 80
    timer_x = (frame_width - timer_width) // 2
    timer_y = frame_height - timer_height - 20
    cv2.rectangle(overlay, (timer_x, timer_y), (timer_x + timer_width, timer_y + timer_height), (0, 0, 0), -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    minutes = int(seconds_remaining // 60)
    seconds = int(seconds_remaining % 60)
    time_str = f"{minutes:02d}:{seconds:02d}"
    cv2.putText(frame, "Stay Alert For:", (timer_x + 20, timer_y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, time_str, (timer_x + 100, timer_y + 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
    return frame

def detect_drowsiness(frame, face_mesh, ear_threshold, mar_threshold, drowsy_time_seconds, awake_time_seconds, fps):
    frame = prepare_image(frame)
    if frame is None:
        return None, "Error", 0, 0, None, None
    
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
    
    results = face_mesh.process(rgb_frame)
    
    status = "Awake"
    seconds_remaining = None
    debug_info = {}
    
    if not results.multi_face_landmarks:
        cv2.putText(frame, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame, status, 0, 0, seconds_remaining, debug_info
    
    landmarks = []
    for landmark in results.multi_face_landmarks[0].landmark:
        x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
        landmarks.append((x, y))
    
    mp_drawing.draw_landmarks(
        frame, results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
    )
    
    left_eye = [landmarks[i] for i in LEFT_EYE_INDICES]
    right_eye = [landmarks[i] for i in RIGHT_EYE_INDICES]
    
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0
    
    mouth = [landmarks[i] for i in MOUTH_INDICES]
    mar = mouth_aspect_ratio(mouth)
    
    cv2.polylines(frame, [np.array(left_eye)], True, (0, 255, 0), 1)
    cv2.polylines(frame, [np.array(right_eye)], True, (0, 255, 0), 1)
    cv2.polylines(frame, [np.array(mouth)], True, (0, 255, 255), 1)
    
    debug_info = {
        'landmarks': landmarks,
        'left_ear': left_ear,
        'right_ear': right_ear,
        'avg_ear': avg_ear,
        'mar': mar
    }
    
    is_eyes_closed = left_ear < ear_threshold and right_ear < ear_threshold
    is_yawning = mar > mar_threshold
    
    if hasattr(st.session_state, 'prev_drowsy_states'):
        st.session_state.prev_drowsy_states.append(is_eyes_closed)
        if len(st.session_state.prev_drowsy_states) > 5:  # Keep last 5 states
            st.session_state.prev_drowsy_states.pop(0)
        # Only consider drowsy if majority of recent frames show drowsiness
        is_consistently_drowsy = sum(st.session_state.prev_drowsy_states) >= 3
    else:
        st.session_state.prev_drowsy_states = [is_eyes_closed]
        is_consistently_drowsy = is_eyes_closed
    
    if 'drowsy_count' not in st.session_state:
        st.session_state.drowsy_count = 0
    if 'awake_count' not in st.session_state:
        st.session_state.awake_count = 0
    
    drowsy_count = st.session_state.drowsy_count
    awake_count = st.session_state.awake_count
    
    status = "DROWSY!" if drowsy_count >= drowsy_time_seconds * fps else "Awake"
    
    if is_consistently_drowsy or is_yawning:
        drowsy_count += 1
        awake_count = 0
        
        if drowsy_count >= drowsy_time_seconds * fps:
            status = "DROWSY!"
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if not mixer.music.get_busy():
                play_alarm()
    else:
        if status == "DROWSY!" or mixer.music.get_busy():  # Check if alarm is playing
            awake_count += 1
            frames_remaining = awake_time_seconds * fps - awake_count
            seconds_remaining = max(0, frames_remaining / fps)
            
            if seconds_remaining > 0:
                frame = draw_timer(frame, seconds_remaining, frame_width, frame_height)
            
            if awake_count >= awake_time_seconds * fps:
                status = "Awake"
                drowsy_count = 0
                stop_alarm()
        else:
            drowsy_count = 0
            awake_count = 0
    
    st.session_state.drowsy_count = drowsy_count
    st.session_state.awake_count = awake_count
    
    cv2.putText(frame, f"EAR: {avg_ear:.2f}", (frame_width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"MAR: {mar:.2f}", (frame_width - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if status == "Awake" else (0, 0, 255), 2)
    
    return frame, status, drowsy_count, awake_count, seconds_remaining, debug_info

def main():
    if 'ear_threshold' not in st.session_state:
        st.session_state.ear_threshold = 0.25
    if 'mar_threshold' not in st.session_state:
        st.session_state.mar_threshold = 0.6
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'calibrating' not in st.session_state:
        st.session_state.calibrating = False
    if 'prev_drowsy_states' not in st.session_state:
        st.session_state.prev_drowsy_states = []
    
    st.title("Advanced Drowsiness Detection System")
    st.markdown("""
    This system monitors drowsiness using eye closure, yawning, and head pose analysis.
    Adjust parameters and calibrate for accurate detection.
    """)
    
    st.sidebar.header("Detection Parameters")
    ear_threshold = st.sidebar.slider("EAR Threshold", 0.1, 0.4, 0.25, 0.01)
    mar_threshold = st.sidebar.slider("MAR Threshold (Yawn)", 0.4, 0.8, 0.6, 0.01)
    drowsy_time_seconds = st.sidebar.slider("Drowsy Detection Time (seconds)", 0.5, 5.0, 1.0, 0.1)
    awake_time_seconds = st.sidebar.slider("Recovery Time (seconds)", 5, 30, 5, 1)
    frame_width = st.sidebar.slider("Camera Width", 300, 800, 640)
    
    debug_mode = st.sidebar.checkbox("Debug Mode", value=True)
    calibrate_button = st.sidebar.button("Calibrate Thresholds")
    
    sensitivity = st.sidebar.slider("Detection Sensitivity", 0.5, 1.5, 0.8, 0.1)
    adjusted_ear_threshold = st.session_state.ear_threshold * sensitivity
    
    if not os.path.isfile("music.wav"):
        st.warning("Warning: 'music.wav' not found. Place an alarm sound file in the directory.")
    
    video_placeholder = st.empty()
    status_placeholder = st.empty()
    timer_placeholder = st.empty()
    debug_placeholder = st.empty()
    
    col1, col2 = st.columns(2)
    start_button = col1.button("Start Detection")
    stop_button = col2.button("Stop Detection")
    
    if start_button:
        st.session_state.running = True
    if stop_button:
        st.session_state.running = False
        stop_alarm()
        st.rerun()
    
    if calibrate_button:
        st.session_state.calibrating = True
    
    if st.session_state.get('calibrating', False):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
            return
        
        with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            ear_open, ear_closed, mar_rest, mar_yawn = [], [], [], []
            st.write("Calibrating... Open your eyes wide for 5 seconds.")
            start_time = time.time()
            while time.time() - start_time < 5:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (frame_width, int(frame_width * frame.shape[0] / frame.shape[1])))
                frame, _, _, _, _, debug_info = detect_drowsiness(frame, face_mesh, 0.3, 0.5, 1.0, 5, 10)
                if frame is not None and debug_info and 'landmarks' in debug_info:
                    landmarks = debug_info['landmarks']
                    avg_ear = (eye_aspect_ratio([landmarks[i] for i in LEFT_EYE_INDICES]) + 
                              eye_aspect_ratio([landmarks[i] for i in RIGHT_EYE_INDICES])) / 2.0
                    ear_open.append(avg_ear)
                    video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            st.write("Close your eyes for 5 seconds.")
            start_time = time.time()
            while time.time() - start_time < 5:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (frame_width, int(frame_width * frame.shape[0] / frame.shape[1])))
                frame, _, _, _, _, debug_info = detect_drowsiness(frame, face_mesh, 0.3, 0.5, 1.0, 5, 10)
                if frame is not None and debug_info and 'landmarks' in debug_info:
                    landmarks = debug_info['landmarks']
                    avg_ear = (eye_aspect_ratio([landmarks[i] for i in LEFT_EYE_INDICES]) + 
                              eye_aspect_ratio([landmarks[i] for i in RIGHT_EYE_INDICES])) / 2.0
                    ear_closed.append(avg_ear)
                    video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            st.write("Yawn widely for 5 seconds.")
            start_time = time.time()
            while time.time() - start_time < 5:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (frame_width, int(frame_width * frame.shape[0] / frame.shape[1])))
                frame, _, _, _, _, debug_info = detect_drowsiness(frame, face_mesh, 0.3, 0.5, 1.0, 5, 10)
                if frame is not None and debug_info and 'landmarks' in debug_info:
                    landmarks = debug_info['landmarks']
                    mar = mouth_aspect_ratio([landmarks[i] for i in MOUTH_INDICES])
                    mar_yawn.append(mar)
                    video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            st.write("Rest your mouth for 5 seconds.")
            start_time = time.time()
            while time.time() - start_time < 5:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (frame_width, int(frame_width * frame.shape[0] / frame.shape[1])))
                frame, _, _, _, _, debug_info = detect_drowsiness(frame, face_mesh, 0.3, 0.5, 1.0, 5, 10)
                if frame is not None and debug_info and 'landmarks' in debug_info:
                    landmarks = debug_info['landmarks']
                    mar = mouth_aspect_ratio([landmarks[i] for i in MOUTH_INDICES])
                    mar_rest.append(mar)
                    video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            if ear_open and ear_closed and mar_rest and mar_yawn:
                st.session_state.ear_threshold = (min(ear_open) + max(ear_closed)) / 2
                st.session_state.mar_threshold = (min(mar_yawn) + max(mar_rest)) / 2
                st.success(f"Calibration complete! EAR Threshold: {st.session_state.ear_threshold:.2f}, MAR Threshold: {st.session_state.mar_threshold:.2f}")
            else:
                st.error("Calibration failed. Ensure face is detected.")
            cap.release()
            st.session_state.calibrating = False
    
    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
            return
        
        with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
            prev_time = time.time()
            while cap.isOpened() and st.session_state.running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    st.error("Failed to grab frame.")
                    break
                
                frame = cv2.resize(frame, (frame_width, int(frame_width * frame.shape[0] / frame.shape[1])))
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 10
                prev_time = curr_time
                
                frame, status, drowsy_count, awake_count, seconds_remaining, debug_info = detect_drowsiness(
                    frame, face_mesh, adjusted_ear_threshold, st.session_state.mar_threshold, 
                    drowsy_time_seconds, awake_time_seconds, fps
                )
                
                if frame is None:
                    continue
                
                if status == "DROWSY!":
                    status_placeholder.error(f"Status: {status} - Drowsy for {drowsy_count/fps:.1f}s")
                    if seconds_remaining is not None:
                        timer_placeholder.warning(f"Stay alert for {seconds_remaining:.1f}s to stop alarm")
                    else:
                        timer_placeholder.empty()
                else:
                    if awake_count > 0:
                        status_placeholder.warning(f"Status: Recovering - Awake for {awake_count/fps:.1f}s")
                        timer_placeholder.info(f"Stay alert for {seconds_remaining:.1f}s to stop alarm")
                    else:
                        status_placeholder.success(f"Status: {status}")
                        timer_placeholder.empty()
                
                if debug_mode:
                    debug_text = f"FPS: {fps:.1f}\nDrowsy Count: {drowsy_count}\nAwake Count: {awake_count}"
                    debug_placeholder.text(debug_text)
                else:
                    debug_placeholder.empty()
                
                video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            cap.release()
            stop_alarm()

    if st.sidebar.button("Calibrate EAR Threshold"):
        st.session_state.calibrating = True
        st.session_state.ear_open = []
        st.session_state.ear_closed = []
        
    if st.session_state.calibrating:
        
        if len(st.session_state.ear_open) > 0 and len(st.session_state.ear_closed) > 0:
            open_avg = sum(st.session_state.ear_open) / len(st.session_state.ear_open)
            closed_avg = sum(st.session_state.ear_closed) / len(st.session_state.ear_closed)
            # Set threshold between open and closed values
            st.session_state.ear_threshold = (open_avg + closed_avg) / 2
            st.success(f"Calibration complete! EAR threshold set to {st.session_state.ear_threshold:.3f}")

    if st.sidebar.button("Reset Counters"):
        st.session_state.drowsy_count = 0
        st.session_state.awake_count = 0

    if st.sidebar.button("Stop Alarm"):
        stop_alarm()
        st.session_state.drowsy_count = 0
        st.session_state.awake_count = 0

if __name__ == "__main__":
    main()