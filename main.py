"""
ADVANCED DROWSINESS DETECTION SYSTEM - GLASSES COMPATIBLE
Real-time monitoring with enhanced eye detection for glasses wearers
"""

import cv2
import numpy as np
import pygame
import time
from scipy.spatial import distance
from scipy.signal import savgol_filter
import dlib
from collections import deque

# Initialize pygame for audio
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

# Initialize dlib face detection
print("[INFO] Loading dlib models...")
detector = dlib.get_frontal_face_detector()

try:
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    print("[✓] Models loaded successfully")
except:
    print("ERROR: Download shape_predictor_68_face_landmarks.dat")
    print("URL: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

# Facial landmark indices
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

# Enhanced thresholds with adaptive adjustment
EAR_CRITICAL = 0.20   # Eyes fully closed
EAR_DROWSY = 0.25     # Eyes half-closed
EAR_NORMAL = 0.28     # Eyes open
BLINK_FRAMES = 3      # Max frames for a normal blink (0.1s at 30fps)

# Data buffers
ear_history = deque(maxlen=150)  # 5 seconds of history at 30fps
ear_buffer = deque(maxlen=5)

# System state
frame_count = 0
blink_count = 0
alert_count = 0
closed_frames = 0
alarm_active = False
last_ear = 0.3
eyes_open = True
close_start_frame = 0
ear_baseline = 0.30
calibrated = False

# Glasses detection state
glasses_detected = False
glasses_check_interval = 30
glasses_check_counter = 0

# Performance tracking
fps = 0
fps_time = time.time()
fps_count = 0

def detect_glasses(frame, face_rect, landmarks):
    """
    Detect if person is wearing glasses using edge detection
    Returns: (has_glasses: bool, confidence: float)
    """
    try:
        # Extract eye regions with padding
        left_eye_points = landmarks[LEFT_EYE]
        right_eye_points = landmarks[RIGHT_EYE]
        
        # Get bounding box for both eyes with extra padding for frames
        all_eye_points = np.vstack([left_eye_points, right_eye_points])
        x_min = max(0, np.min(all_eye_points[:, 0]) - 25)
        x_max = min(frame.shape[1], np.max(all_eye_points[:, 0]) + 25)
        y_min = max(0, np.min(all_eye_points[:, 1]) - 20)
        y_max = min(frame.shape[0], np.max(all_eye_points[:, 1]) + 20)
        
        eye_region = frame[y_min:y_max, x_min:x_max]
        
        if eye_region.size == 0:
            return False, 0.0
        
        # Convert to grayscale if needed
        if len(eye_region.shape) == 3:
            eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        else:
            eye_gray = eye_region
        
        # Apply Gaussian blur to reduce noise
        eye_blur = cv2.GaussianBlur(eye_gray, (5, 5), 0)
        
        # Edge detection for glass frames
        edges = cv2.Canny(eye_blur, 30, 100)
        
        # Check for horizontal lines (typical of glasses frames)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Count edge pixels
        edge_density = np.sum(horizontal_lines > 0) / horizontal_lines.size
        
        # Check for strong vertical edges (nose bridge area)
        mid_x = eye_region.shape[1] // 2
        bridge_region = edges[:, max(0, mid_x-10):min(eye_region.shape[1], mid_x+10)]
        bridge_density = np.sum(bridge_region > 0) / bridge_region.size if bridge_region.size > 0 else 0
        
        # Detect reflections (common with glasses)
        _, bright_spots = cv2.threshold(eye_blur, 200, 255, cv2.THRESH_BINARY)
        reflection_ratio = np.sum(bright_spots > 0) / bright_spots.size
        
        # Combined confidence score
        confidence = (edge_density * 2.0) + (bridge_density * 1.5) + (reflection_ratio * 1.0)
        
        # Threshold for glasses detection
        has_glasses = confidence > 0.15
        
        return has_glasses, confidence
        
    except Exception as e:
        return False, 0.0

def preprocess_eye_region(frame, eye_landmarks, has_glasses=False):
    """
    Enhanced preprocessing for eye region, especially for glasses
    """
    try:
        # Get eye bounding box
        x_coords = eye_landmarks[:, 0]
        y_coords = eye_landmarks[:, 1]
        
        x_min = max(0, np.min(x_coords) - 5)
        x_max = min(frame.shape[1], np.max(x_coords) + 5)
        y_min = max(0, np.min(y_coords) - 5)
        y_max = min(frame.shape[0], np.max(y_coords) + 5)
        
        eye_roi = frame[y_min:y_max, x_min:x_max]
        
        if eye_roi.size == 0:
            return None
        
        if has_glasses:
            # Enhanced processing for glasses
            # 1. Reduce glare and reflections
            eye_lab = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(eye_lab)
            
            # Apply CLAHE to L channel to reduce glare
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
            l_enhanced = clahe.apply(l)
            
            # Merge and convert back
            eye_enhanced = cv2.merge([l_enhanced, a, b])
            eye_roi = cv2.cvtColor(eye_enhanced, cv2.COLOR_LAB2BGR)
            
            # 2. Bilateral filter to preserve edges while smoothing
            eye_roi = cv2.bilateralFilter(eye_roi, 5, 50, 50)
        
        return eye_roi
        
    except Exception as e:
        return None

def calculate_ear_robust(eye_landmarks, has_glasses=False):
    """
    Enhanced EAR calculation with outlier handling for glasses
    """
    try:
        # Standard EAR calculation
        v1 = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        v2 = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        h = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        if h == 0:
            return 0.3
        
        ear = (v1 + v2) / (2.0 * h)
        
        if has_glasses:
            # Additional validation for glasses
            # Check if landmarks are reasonable
            eye_width = h
            eye_height = (v1 + v2) / 2
            
            # Aspect ratio should be reasonable (eyes are wider than tall)
            if eye_width < eye_height * 1.5:
                # Possibly bad detection, use previous value
                return None
            
            # Clamp extreme values that might be from reflections
            ear = max(0.10, min(0.45, ear))
        
        return ear
        
    except Exception as e:
        return None

def smooth_ear_value(ear_value, has_glasses=False):
    """Apply smoothing filter with enhanced filtering for glasses"""
    if ear_value is None:
        # Use last valid value if current is invalid
        if len(ear_buffer) > 0:
            return ear_buffer[-1]
        return 0.3
    
    ear_buffer.append(ear_value)
    
    if len(ear_buffer) >= 5:
        values = list(ear_buffer)
        
        if has_glasses:
            # More aggressive smoothing for glasses
            # Remove outliers
            mean_val = np.mean(values)
            std_val = np.std(values)
            filtered_values = [v for v in values if abs(v - mean_val) < 2 * std_val]
            
            if len(filtered_values) >= 3:
                return savgol_filter(filtered_values, len(filtered_values), min(2, len(filtered_values)-1))[-1]
        
        return savgol_filter(values, 5, 2)[-1]
    
    return ear_value

def create_alarm_sound():
    """Generate urgent alarm sound"""
    try:
        sample_rate = 44100
        duration = 0.8
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Multi-frequency urgent tone
        freq1 = 800 + 400 * np.sin(2 * np.pi * 3 * t)
        freq2 = 1200 + 200 * np.sin(2 * np.pi * 5 * t)
        
        wave1 = np.sin(2 * np.pi * freq1 * t)
        wave2 = np.sin(2 * np.pi * freq2 * t)
        
        # Combine with tremolo effect
        tremolo = 1.0 + 0.5 * np.sin(2 * np.pi * 8 * t)
        combined = (0.6 * wave1 + 0.4 * wave2) * tremolo
        
        # Normalize
        combined = combined / np.max(np.abs(combined))
        audio = (combined * 32767 * 0.85).astype(np.int16)
        
        return pygame.sndarray.make_sound(np.column_stack((audio, audio)))
    except Exception as e:
        print(f"[WARNING] Alarm creation failed: {e}")
        return None

def extract_landmarks(dlib_shape):
    """Convert dlib shape to numpy array"""
    return np.array([(dlib_shape.part(i).x, dlib_shape.part(i).y) 
                     for i in range(68)], dtype=np.int32)

def draw_graph(canvas, x, y, w, h, data, title, color, threshold_lines=None):
    """Draw real-time graph with thresholds"""
    # Background
    cv2.rectangle(canvas, (x, y), (x+w, y+h), (30, 30, 30), -1)
    cv2.rectangle(canvas, (x, y), (x+w, y+h), (60, 60, 60), 2)
    
    # Title
    cv2.putText(canvas, title, (x+10, y+25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    if len(data) < 2:
        return
    
    # Graph area
    graph_padding = 40
    graph_x = x + graph_padding
    graph_y = y + 40
    graph_w = w - 2 * graph_padding
    graph_h = h - 60
    
    # Grid
    for i in range(5):
        y_pos = graph_y + int(i * graph_h / 4)
        cv2.line(canvas, (graph_x, y_pos), (graph_x + graph_w, y_pos), 
                (50, 50, 50), 1)
    
    # Threshold lines
    if threshold_lines:
        for threshold, line_color, label in threshold_lines:
            y_pos = graph_y + graph_h - int((threshold - 0.1) / 0.3 * graph_h)
            if graph_y <= y_pos <= graph_y + graph_h:
                cv2.line(canvas, (graph_x, y_pos), (graph_x + graph_w, y_pos), 
                        line_color, 2, cv2.LINE_AA)
                cv2.putText(canvas, label, (graph_x + graph_w + 5, y_pos + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, line_color, 1)
    
    # Plot data
    data_list = list(data)
    points = []
    
    for i, value in enumerate(data_list):
        normalized = (value - 0.1) / 0.3
        normalized = max(0, min(1, normalized))
        
        px = graph_x + int(i * graph_w / len(data_list))
        py = graph_y + graph_h - int(normalized * graph_h)
        points.append((px, py))
    
    # Draw line
    for i in range(len(points) - 1):
        cv2.line(canvas, points[i], points[i+1], color, 2, cv2.LINE_AA)
    
    # Current value
    if data_list:
        current = data_list[-1]
        cv2.putText(canvas, f"{current:.3f}", (x + 10, y + h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def draw_status_indicator(canvas, x, y, w, h, status, status_color, additional_info=""):
    """Draw large status indicator"""
    cv2.rectangle(canvas, (x, y), (x+w, y+h), (25, 25, 25), -1)
    cv2.rectangle(canvas, (x, y), (x+w, y+h), status_color, 5)
    
    # Status text
    font_scale = 1.2
    thickness = 3
    text_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = x + (w - text_size[0]) // 2
    text_y = y + (h + text_size[1]) // 2
    
    cv2.putText(canvas, status, (text_x, text_y),
               cv2.FONT_HERSHEY_SIMPLEX, font_scale, status_color, thickness)
    
    # Additional info
    if additional_info:
        cv2.putText(canvas, additional_info, (x + 20, y + h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

def draw_metric_card(canvas, x, y, w, h, label, value, color):
    """Draw compact metric card"""
    cv2.rectangle(canvas, (x, y), (x+w, y+h), (35, 35, 35), -1)
    cv2.rectangle(canvas, (x, y), (x+w, y+h), color, 3)
    
    cv2.putText(canvas, label, (x+20, y+35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
    cv2.putText(canvas, str(value), (x+20, y+75),
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

# Initialize alarm
alarm_sound = create_alarm_sound()
if alarm_sound:
    print("[✓] Alarm system ready")

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Verify camera
ret, test_frame = cap.read()
if ret:
    actual_h, actual_w = test_frame.shape[:2]
    print(f"[INFO] Camera active: {actual_w}x{actual_h}")
else:
    print("[ERROR] Camera initialization failed")
    exit()

print("\n" + "="*70)
print(" DROWSINESS DETECTION SYSTEM - ACTIVE".center(70))
print("="*70)
print(" Calibrating... Keep eyes open and look at camera")
print("="*70 + "\n")

# System settings
sound_enabled = True
cal_start_time = time.time()
cal_buffer = deque(maxlen=60)

# Face detection optimization
detect_interval = 2
last_detected_face = None
detect_counter = 0

# Display settings
DISPLAY_W, DISPLAY_H = 1600, 900

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Frame capture failed")
        break
    
    frame_count += 1
    fps_count += 1
    
    # Calculate FPS
    if time.time() - fps_time >= 1.0:
        fps = fps_count
        fps_count = 0
        fps_time = time.time()
    
    # Mirror frame
    frame = cv2.flip(frame, 1)
    
    # Create display canvas
    display = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    display[:] = (15, 15, 15)
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Enhanced preprocessing for glasses
    # Apply CLAHE to handle reflections better
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Default status
    status = "NO FACE DETECTED"
    status_color = (100, 100, 100)
    status_info = ""
    
    # Optimized face detection
    detect_counter += 1
    if detect_counter >= detect_interval or last_detected_face is None:
        faces = detector(gray, 0)
        if len(faces) > 0:
            last_detected_face = faces[0]
        detect_counter = 0
    
    # Process if face detected
    if last_detected_face is not None:
        try:
            shape = predictor(gray, last_detected_face)
            landmarks = extract_landmarks(shape)
            
            # Periodic glasses detection
            glasses_check_counter += 1
            if glasses_check_counter >= glasses_check_interval:
                glasses_detected, glasses_conf = detect_glasses(frame, last_detected_face, landmarks)
                glasses_check_counter = 0
                if glasses_detected:
                    print(f"[INFO] Glasses detected (confidence: {glasses_conf:.3f})")
            
            # Get eye landmarks
            left_eye = landmarks[LEFT_EYE]
            right_eye = landmarks[RIGHT_EYE]
            
            # Calculate EAR with glasses handling
            left_ear = calculate_ear_robust(left_eye, glasses_detected)
            right_ear = calculate_ear_robust(right_eye, glasses_detected)
            
            # Handle invalid readings
            if left_ear is not None and right_ear is not None:
                avg_ear = (left_ear + right_ear) / 2.0
                smoothed_ear = smooth_ear_value(avg_ear, glasses_detected)
                last_ear = smoothed_ear
            elif left_ear is not None:
                smoothed_ear = smooth_ear_value(left_ear, glasses_detected)
                last_ear = smoothed_ear
            elif right_ear is not None:
                smoothed_ear = smooth_ear_value(right_ear, glasses_detected)
                last_ear = smoothed_ear
            else:
                # Both invalid, use last known value
                smoothed_ear = last_ear
            
            # Store history
            ear_history.append(smoothed_ear)
            
            # Calibration phase
            if not calibrated:
                elapsed = time.time() - cal_start_time
                if elapsed < 3.0:
                    if smoothed_ear > 0.22:
                        cal_buffer.append(smoothed_ear)
                    status = "CALIBRATING..."
                    status_color = (0, 200, 255)
                    status_info = f"{3 - int(elapsed)}s remaining"
                else:
                    if len(cal_buffer) > 20:
                        ear_baseline = np.mean(list(cal_buffer))
                        calibrated = True
                        glasses_status = " (Glasses Mode)" if glasses_detected else ""
                        print(f"[✓] Calibration complete{glasses_status}! Baseline EAR: {ear_baseline:.3f}\n")
                    else:
                        cal_start_time = time.time()
                        cal_buffer.clear()
            
            # Draw face detection box
            fx = last_detected_face.left()
            fy = last_detected_face.top()
            fw = last_detected_face.width()
            fh = last_detected_face.height()
            
            box_color = (0, 200, 255) if glasses_detected else (0, 255, 0)
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), box_color, 2)
            
            # Draw eye contours
            cv2.polylines(frame, [left_eye], True, (0, 255, 255), 1)
            cv2.polylines(frame, [right_eye], True, (0, 255, 255), 1)
            
            # Add glasses indicator
            if glasses_detected:
                cv2.putText(frame, "GLASSES", (fx, fy-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            
            # Eye state analysis (only if calibrated)
            if calibrated:
                if smoothed_ear <= EAR_CRITICAL:
                    if eyes_open:
                        close_start_frame = frame_count
                        eyes_open = False
                    
                    closed_frames = frame_count - close_start_frame
                    
                    if closed_frames > BLINK_FRAMES:
                        status = "EYES CLOSED ALERT"
                        status_color = (0, 0, 255)
                        status_info = f"Closed for {closed_frames} frames"
                        
                        # Immediate alarm
                        if not alarm_active and alarm_sound and sound_enabled:
                            alarm_sound.play(-1)
                            alarm_active = True
                            alert_count += 1
                    else:
                        status = "BLINKING"
                        status_color = (255, 200, 0)
                
                elif smoothed_ear <= EAR_DROWSY:
                    status = "DROWSY"
                    status_color = (0, 255, 255)
                    status_info = "Eyes partially closed"
                    
                    if not eyes_open:
                        duration = frame_count - close_start_frame
                        if duration <= BLINK_FRAMES:
                            blink_count += 1
                        eyes_open = True
                        closed_frames = 0
                    
                    if alarm_active and alarm_sound:
                        alarm_sound.stop()
                        alarm_active = False
                
                else:  # EAR is good
                    if not eyes_open:
                        duration = frame_count - close_start_frame
                        if duration <= BLINK_FRAMES:
                            blink_count += 1
                        eyes_open = True
                        closed_frames = 0
                    
                    status = "OK"
                    status_color = (0, 255, 0)
                    status_info = "Alert and focused"
                    
                    if alarm_active and alarm_sound:
                        alarm_sound.stop()
                        alarm_active = False
            
        except Exception as e:
            print(f"[WARNING] Processing error: {e}")
            last_detected_face = None
    else:
        # No face detected - reset
        if not eyes_open:
            eyes_open = True
            closed_frames = 0
            if alarm_active and alarm_sound:
                alarm_sound.stop()
                alarm_active = False
    
    # === LAYOUT === 
    
    # Video feed (left side)
    video_x, video_y = 20, 20
    video_w, video_h = 800, 600
    video_frame = cv2.resize(frame, (video_w, video_h))
    display[video_y:video_y+video_h, video_x:video_x+video_w] = video_frame
    
    # Add alert border to video
    if status == "EYES CLOSED ALERT":
        border_thickness = 8 if frame_count % 10 < 5 else 12
        cv2.rectangle(display, (video_x-5, video_y-5), 
                     (video_x+video_w+5, video_y+video_h+5), 
                     status_color, border_thickness)
    elif status == "DROWSY":
        cv2.rectangle(display, (video_x-3, video_y-3), 
                     (video_x+video_w+3, video_y+video_h+3), 
                     status_color, 4)
    
    # Status indicator (below video)
    status_y = video_y + video_h + 20
    draw_status_indicator(display, video_x, status_y, video_w, 120, 
                         status, status_color, status_info)
    
    # Metrics cards (below status)
    metrics_y = status_y + 140
    card_w = 250
    card_h = 100
    card_gap = 25
    
    draw_metric_card(display, video_x, metrics_y, card_w, card_h, 
                    "BLINKS", blink_count, (0, 255, 0))
    draw_metric_card(display, video_x + card_w + card_gap, metrics_y, card_w, card_h,
                    "ALERTS", alert_count, (0, 0, 255))
    draw_metric_card(display, video_x + 2*(card_w + card_gap), metrics_y, card_w, card_h,
                    "EAR", f"{last_ear:.3f}", status_color)
    
    # Graph (right side)
    graph_x = 860
    graph_y = 20
    graph_w = 720
    graph_h = 840
    
    threshold_lines = [
        (EAR_CRITICAL, (0, 0, 255), "Critical"),
        (EAR_DROWSY, (0, 255, 255), "Drowsy"),
        (EAR_NORMAL, (0, 255, 0), "Normal")
    ]
    
    draw_graph(display, graph_x, graph_y, graph_w, graph_h, 
              ear_history, "EYE ASPECT RATIO - REAL TIME", 
              (100, 200, 255), threshold_lines)
    
    # Bottom info bar
    bar_h = 40
    cv2.rectangle(display, (0, DISPLAY_H-bar_h), (DISPLAY_W, DISPLAY_H), (20, 20, 20), -1)
    
    glasses_text = " | Glasses: YES" if glasses_detected else ""
    info_text = f"FPS: {fps}  |  Sound: {'ON' if sound_enabled else 'OFF'}  |  " \
                f"Status: {'Calibrated' if calibrated else 'Calibrating'}{glasses_text}"
    cv2.putText(display, info_text, (20, DISPLAY_H-15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.putText(display, "Q:Quit | R:Reset | S:Sound | C:Recalibrate | SPACE:Test Alarm", 
               (DISPLAY_W-580, DISPLAY_H-15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    
    # Display
    cv2.imshow("Drowsiness Detection System", display)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('r'):
        blink_count = alert_count = closed_frames = 0
        eyes_open = True
        ear_history.clear()
        ear_buffer.clear()
        if alarm_active and alarm_sound:
            alarm_sound.stop()
            alarm_active = False
        print("[SYSTEM RESET]")
    elif key == ord('s'):
        sound_enabled = not sound_enabled
        if not sound_enabled and alarm_active and alarm_sound:
            alarm_sound.stop()
            alarm_active = False
        print(f"[SOUND] {'ENABLED' if sound_enabled else 'DISABLED'}")
    elif key == ord('c'):
        calibrated = False
        cal_buffer.clear()
        cal_start_time = time.time()
        glasses_check_counter = 0  # Re-check for glasses
        print("[RECALIBRATING...]")
    elif key == 32:  # Space
        if alarm_sound:
            print("[TESTING ALARM]")
            alarm_sound.play()
            time.sleep(1.0)
            alarm_sound.stop()

# Cleanup
print("\n" + "="*70)
print(" SESSION SUMMARY".center(70))
print("="*70)
print(f" Total Frames: {frame_count}")
print(f" Blinks Detected: {blink_count}")
print(f" Alerts Triggered: {alert_count}")
print(f" Glasses Mode: {'YES' if glasses_detected else 'NO'}")
print("="*70)

if alarm_sound:
    alarm_sound.stop()
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
print("\n[✓] System shutdown complete")