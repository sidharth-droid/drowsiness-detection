# Drowsiness Detection System

A real-time drowsiness detection application that uses computer vision and facial landmarks to monitor alertness and prevent accidents due to drowsiness.

## Features

- Real-time eye closure detection
- Yawning detection
- Head pose analysis
- Customizable sensitivity settings
- Audio alerts when drowsiness is detected
- Recovery timer display
- Calibration for personalized thresholds
- Debug mode for monitoring metrics

## Demo

![Drowsiness Detection Demo](assets/demo.gif)

## Requirements

- Python 3.10+
- Webcam
- Sound output device

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sidharth-droid/drowsiness-detection.git
   cd drowsiness-detection
   ```

2. Create a virtual environment:
   ```bash
   python -m venv drowsiness_virtual
   ```

3. Activate the virtual environment:
   - Windows:
     ```bash
     drowsiness_virtual\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source drowsiness_virtual/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Download the alert sound:
   - Place a WAV file named `music.wav` in the project root directory
   - Or use your own sound file and update the filename in the code

## Usage

1. Run the application:
   ```bash
   streamlit run Drowsiness.py
   ```

2. Allow camera access when prompted

3. Use the sidebar to adjust parameters:
   - EAR Threshold: Eye Aspect Ratio threshold for detecting closed eyes
   - MAR Threshold: Mouth Aspect Ratio threshold for detecting yawns
   - Drowsy Detection Time: How long eyes must be closed to trigger an alert
   - Recovery Time: How long eyes must be open to stop the alert
   - Detection Sensitivity: Adjust to fine-tune detection

4. Click "Start Detection" to begin monitoring

5. Optional: Use "Calibrate Thresholds" to personalize detection thresholds:
   - Follow on-screen instructions to calibrate with your eyes open, closed, and mouth positions
   - This improves accuracy for your specific facial features

6. If the alarm triggers, keep your eyes open for the recovery time to stop it

7. Use "Stop Alarm" button to manually stop the alarm if needed

## How It Works

### Technical Architecture

The system uses a multi-modal approach to detect drowsiness through three key indicators:

1. **Eye Closure Detection**:
   - Uses the Eye Aspect Ratio (EAR) to measure eye openness
   - Compares EAR against a threshold to determine if eyes are closed
   - Requires consistent detection across multiple frames to reduce false positives

2. **Yawn Detection**:
   - Uses the Mouth Aspect Ratio (MAR) to detect mouth openness
   - Identifies yawning when MAR exceeds a threshold

3. **Head Pose Analysis**:
   - Monitors head position to detect nodding or drooping
   - Uses facial landmarks to estimate head orientation

### Models and Algorithms

1. **MediaPipe Face Mesh**:
   - Provides 468 facial landmarks with high precision
   - Used for accurate facial feature tracking
   - More robust than traditional face detection in varying lighting conditions

2. **Eye Aspect Ratio (EAR)**:
   - Calculates the ratio of eye height to width
   - Formula: EAR = (A + B) / (2.0 * C) where:
     - A, B are vertical eye landmark distances
     - C is horizontal eye landmark distance
   - Lower values indicate more closed eyes

3. **Mouth Aspect Ratio (MAR)**:
   - Similar to EAR but applied to mouth landmarks
   - Higher values indicate open mouth/yawning

4. **Temporal Consistency Check**:
   - Uses a sliding window of recent frames to ensure consistent detection
   - Reduces false positives from blinking or momentary movements

### Code Structure

- `prepare_image()`: Ensures images are in the correct format for processing
- `eye_aspect_ratio()`: Calculates EAR from eye landmarks
- `mouth_aspect_ratio()`: Calculates MAR from mouth landmarks
- `head_pose_estimation()`: Analyzes head position for nodding detection
- `draw_timer()`: Creates visual countdown timer on screen
- `detect_drowsiness()`: Main detection function that combines all indicators
- `main()`: Streamlit UI and application flow control

## Customization

### Adjusting Sensitivity

- Increase EAR threshold to make eye closure detection more sensitive
- Decrease MAR threshold to make yawn detection more sensitive
- Reduce drowsy detection time for faster alerts
- Use the calibration feature for personalized thresholds

### Adding New Features

The modular code structure makes it easy to add new detection methods:

1. Create a new detection function
2. Add the function call in `detect_drowsiness()`
3. Update the UI in `main()` to include controls for the new feature

## Troubleshooting

### Common Issues

1. **Camera Not Working**:
   - Ensure your webcam is connected and not in use by another application
   - Try changing the camera index in `cv2.VideoCapture(0)` to `1` or `2`

2. **No Face Detected**:
   - Improve lighting conditions
   - Position yourself directly in front of the camera
   - Ensure nothing is obstructing your face

3. **False Positives/Negatives**:
   - Use the calibration feature to set personalized thresholds
   - Adjust sensitivity settings
   - Enable debug mode to monitor detection metrics

4. **Sound Not Playing**:
   - Ensure `music.wav` exists in the project directory
   - Check your system sound settings
   - Use the "Stop Alarm" button to reset if stuck

5. **NumPy Version Issues**:
   - If you encounter "Unsupported image type" errors, downgrade NumPy:
     ```bash
     pip install numpy==1.23.5
     ```

6. **Dlib Installation Issues**:
   - If you encounter errors installing dlib, use the pre-built wheel included in the repository:
     ```bash
     pip install dlib-19.22.99-cp310-cp310-win_amd64.whl
     ```
   - This wheel is compatible with Python 3.10 on Windows. For other platforms, you may need to build dlib from source or find an appropriate wheel.

## Dependencies

- OpenCV: Computer vision and image processing
- MediaPipe: Facial landmark detection
- Streamlit: Web interface
- NumPy: Numerical operations
- SciPy: Scientific computing
- Pygame: Audio playback

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe team for the face mesh model
- Streamlit for the interactive web framework
- The computer vision community for drowsiness detection research

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
