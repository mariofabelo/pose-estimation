# Simple Bicep Curl Demo

A real-time bicep curl tracking application using MediaPipe pose estimation to count repetitions and provide visual feedback.

![Example Curl Tracking](example%20curl%20tracking%20mediapipe.png)

## Features

- **Real-time pose detection** using MediaPipe
- **Automatic rep counting** for both left and right arms
- **Visual feedback** with colored landmarks
- **SF Pro font rendering** for clean, modern UI
- **Dual-arm tracking** with independent counting
- **Configurable parameters** for different exercise standards

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- NumPy
- PIL (Pillow)

## Installation

1. Install the required dependencies:
```bash
pip install opencv-python mediapipe numpy pillow
```

2. Ensure you have the `bicep_curl.py` module in the same directory.

## Usage

Run the demo:
```bash
python simple_bicep_curl_demo.py
```

### Controls
- Press `q` to quit the application

### Visual Indicators

The application displays:
- **Rep counts** for both arms in large, yellow text
- **Colored landmarks**:
  - 🔵 Blue circles: Shoulders
  - 🟢 Green circles: Elbows  
  - 🔴 Red circles: Wrists
- **Pose connections** showing the full skeleton

## Configuration

You can adjust these parameters at the top of the script:

```python
IDEAL_ANGLE = 90      # degrees, for a good curl (arm bent at 90°)
TOLERANCE = 20        # degrees, allowed deviation
EXTENDED_THRESHOLD = 160  # degrees, arm considered straight
```

### Parameter Explanation

- **IDEAL_ANGLE**: The target angle for a proper bicep curl (90° = arm bent at right angle)
- **TOLERANCE**: How much deviation from the ideal angle is acceptable
- **EXTENDED_THRESHOLD**: The angle at which the arm is considered fully extended for rep counting

## How It Works

### Rep Counting Logic

1. **Start of rep**: Arm must be extended (angle > 160°) and then begin curling
2. **End of rep**: Arm returns to extended position (angle > 160°)
3. **Counting**: Each complete cycle (extended → curled → extended) counts as one rep

### Pose Detection

The application uses MediaPipe's pose estimation to track:
- Left shoulder (landmark 11)
- Left elbow (landmark 13) 
- Left wrist (landmark 15)
- Right shoulder (landmark 12)
- Right elbow (landmark 14)
- Right wrist (landmark 16)

### Angle Calculation

The elbow angle is calculated using the three-point method:
- Point A: Shoulder
- Point B: Elbow (vertex)
- Point C: Wrist

The angle at the elbow is computed using vector mathematics.

## Technical Details

### Dependencies

- **MediaPipe**: Google's framework for building perception pipelines
- **OpenCV**: Computer vision library for image processing
- **NumPy**: Numerical computing for angle calculations
- **PIL**: Python Imaging Library for font rendering

### Font Support

The application attempts to load SF Pro font from macOS system locations:
- `/System/Library/Fonts/SF-Pro-Text-Regular.otf`
- `/System/Library/Fonts/SF-Pro-Text-Bold.otf`
- Falls back to system default fonts if SF Pro is unavailable

### Performance

- Optimized for real-time processing
- Configurable detection confidence thresholds
- Efficient landmark tracking with MediaPipe

## Troubleshooting

### Common Issues

1. **"No person detected"**: Ensure you're visible in the camera frame
2. **Poor tracking**: Adjust lighting conditions or camera position
3. **Font errors**: The app will fall back to default fonts if SF Pro is unavailable

### Camera Setup

- Position yourself 2-3 feet from the camera
- Ensure good lighting
- Stand with your side profile visible for best tracking
- Keep both arms visible in the frame

## File Structure

```
├── simple_bicep_curl_demo.py    # Main demo application
├── bicep_curl.py                # BicepCurl class for angle calculations
├── example curl tracking mediapipe.png  # Example screenshot
└── README_simple_bicep_curl_demo.md     # This documentation
```

## Contributing

Feel free to modify the parameters or add new features:
- Adjust rep counting thresholds
- Add audio feedback
- Implement exercise form analysis
- Add data logging capabilities

## License

This project is part of the AI Gym Pose Estimation suite for fitness tracking and analysis.
