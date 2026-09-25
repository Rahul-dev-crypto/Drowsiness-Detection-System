import cv2
import dlib
import numpy as np

print("Testing dlib face detection...")

# Load detector
detector = dlib.get_frontal_face_detector()

# Open camera
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

if not ret:
    print("ERROR: Cannot read from camera")
    exit()

# Try different image formats
print("\n1. Testing with BGR frame...")
try:
    faces = detector(frame, 0)
    print(f"   ✓ BGR works! Detected {len(faces)} faces")
except Exception as e:
    print(f"   ✗ BGR failed: {e}")

print("\n2. Testing with grayscale...")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = np.ascontiguousarray(gray, dtype=np.uint8)
try:
    faces = detector(gray, 0)
    print(f"   ✓ Grayscale works! Detected {len(faces)} faces")
except Exception as e:
    print(f"   ✗ Grayscale failed: {e}")

print("\n3. Testing with RGB...")
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
try:
    faces = detector(rgb, 0)
    print(f"   ✓ RGB works! Detected {len(faces)} faces")
except Exception as e:
    print(f"   ✗ RGB failed: {e}")

print("\nImage properties:")
print(f"Frame shape: {frame.shape}")
print(f"Frame dtype: {frame.dtype}")
print(f"Gray shape: {gray.shape}")
print(f"Gray dtype: {gray.dtype}")
print(f"Is contiguous: {gray.flags['C_CONTIGUOUS']}")

cap.release()
print("\nTest complete!")