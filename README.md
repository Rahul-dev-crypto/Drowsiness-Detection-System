# Drowsiness Detection System

A real-time driver drowsiness detection system using computer vision and facial landmark analysis. Detects eye closure and drowsiness through webcam feed and triggers an audio alarm when the driver appears to be falling asleep. Includes special support for glasses wearers.

## Features

- Real-time Eye Aspect Ratio (EAR) monitoring via webcam
- Automatic glasses detection with enhanced processing to reduce glare/reflection artifacts
- Audio alarm triggered on eye closure beyond normal blink duration
- Auto-calibration on startup to set a personal EAR baseline
- Live EAR graph with critical/drowsy/normal thresholds
- Blink counter and alert counter with session summary on exit

## How It Works

The system uses **dlib's 68-point facial landmark detector** to locate eye landmarks each frame. It calculates the Eye Aspect Ratio (EAR) — a ratio of eye height to width. When eyes are open the EAR stays relatively constant; when closed it drops sharply.

| Threshold | EAR Value | Meaning |
|-----------|-----------|---------|
| Normal | > 0.28 | Eyes open, alert |
| Drowsy | ≤ 0.25 | Eyes half-closed |
| Critical | ≤ 0.20 | Eyes closed — alarm triggers |

For glasses wearers, the system applies CLAHE preprocessing, bilateral filtering, and outlier rejection to handle reflections and distortions from frames.

## Requirements

- Python 3.8+
- Webcam
- `shape_predictor_68_face_landmarks.dat` (see setup below)

## Installation

```bash
pip install -r requirements.txt
```

> Note: `dlib` requires CMake and a C++ compiler. On Windows, install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) before running pip install.

## Download the Landmark Model

The `.dat` file is not included in this repo due to its size (~95MB). Download it from dlib's official source:

```
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```

Extract and place `shape_predictor_68_face_landmarks.dat` in the project root.

## Usage

```bash
python main.py
```

The system will calibrate for 3 seconds on startup — keep your eyes open and look at the camera during this time.

### Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `R` | Reset counters |
| `S` | Toggle sound on/off |
| `C` | Recalibrate |
| `Space` | Test alarm sound |

## Testing

To verify your camera and dlib setup before running the main system:

```bash
python test.py
```

## Project Structure

```
├── main.py                                  # Main detection system
├── test.py                                  # Camera and dlib diagnostic test
├── requirements.txt                         # Python dependencies
└── shape_predictor_68_face_landmarks.dat    # dlib landmark model (download separately)
```

## Dependencies

| Package | Version |
|---------|---------|
| opencv-python | 4.8.1.78 |
| dlib | 19.24.2 |
| scipy | 1.11.4 |
| numpy | 1.24.3 |
| pygame | 2.5.2 |
| imutils | 0.5.4 |
| cmake | 3.27.7 |
