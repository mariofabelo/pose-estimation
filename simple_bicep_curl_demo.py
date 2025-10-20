import cv2
import mediapipe as mp
import numpy as np
from bicep_curl import BicepCurl
from PIL import Image, ImageDraw, ImageFont
import os

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# User-tunable parameters
IDEAL_ANGLE = 90  # degrees, for a good curl (arm bent at 90°)
TOLERANCE = 20    # degrees, allowed deviation
EXTENDED_THRESHOLD = 160  # degrees, arm considered straight

# Load SF Pro font
def load_sf_pro_font(size=24):
    """Load SF Pro font from system or fallback to default"""
    try:
        # Try to load SF Pro from common macOS locations
        font_paths = [
            "/System/Library/Fonts/SF-Pro-Text-Regular.otf",
            "/System/Library/Fonts/SF-Pro-Text-Bold.otf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf"
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, size)
        
        # Fallback to default font
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()

# Initialize font
sf_pro_font = load_sf_pro_font(24)

# Rep counting state
left_rep_count = 0
right_rep_count = 0
left_in_curl = False
right_in_curl = False

def draw_text_with_sf_pro(img, text, org, font_size=24, color=(255, 255, 255), shadow_color=(0, 0, 0), shadow_offset=(2, 2)):
    """Draw text using SF Pro font with PIL"""
    try:
        # Convert OpenCV image to PIL
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # Load font with specified size
        font = load_sf_pro_font(font_size)
        
        # Get text bounding box for positioning
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x, y = org
        
        # Draw shadow
        draw.text((x + shadow_offset[0], y + shadow_offset[1]), text, font=font, fill=shadow_color)
        
        # Draw main text
        draw.text((x, y), text, font=font, fill=color)
        
        # Convert back to OpenCV format
        img_with_text = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img_with_text
        
    except Exception as e:
        print(f"Error drawing text with SF Pro: {e}")
        # Fallback to original OpenCV text
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return img

def draw_text_with_shadow(img, text, org, font, font_scale, color, thickness, shadow_color=(0,0,0), shadow_offset=(2,2)):
    # Draw shadow
    x, y = org
    cv2.putText(img, text, (x + shadow_offset[0], y + shadow_offset[1]), font, font_scale, shadow_color, thickness + 2, cv2.LINE_AA)
    # Draw main text
    cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

def get_landmark_coords(landmarks, landmark_idx, image_shape):
    """Convert MediaPipe landmark to pixel coordinates"""
    landmark = landmarks[landmark_idx]
    h, w = image_shape[:2]
    return (int(landmark.x * w), int(landmark.y * h))

# OpenCV video capture
cap = cv2.VideoCapture(0)

print("Starting Simple Bicep Curl Demo with MediaPipe...")
print("Press 'q' to quit")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection
        results = pose.process(image)
        
        # Draw landmarks
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Get landmark coordinates for bicep curl analysis
            landmarks = results.pose_landmarks.landmark
            
            # MediaPipe pose landmark indices:
            # 11 = left shoulder, 12 = right shoulder
            # 13 = left elbow, 14 = right elbow  
            # 15 = left wrist, 16 = right wrist
            
            try:
                # Get coordinates for left arm
                left_shoulder = get_landmark_coords(landmarks, 11, image.shape)
                left_elbow = get_landmark_coords(landmarks, 13, image.shape)
                left_wrist = get_landmark_coords(landmarks, 15, image.shape)
                
                # Get coordinates for right arm
                right_shoulder = get_landmark_coords(landmarks, 12, image.shape)
                right_elbow = get_landmark_coords(landmarks, 14, image.shape)
                right_wrist = get_landmark_coords(landmarks, 16, image.shape)
                
                # Create BicepCurl instances
                left_curl = BicepCurl(left_shoulder, left_elbow, left_wrist, 
                                    ideal_angle=IDEAL_ANGLE, tolerance=TOLERANCE)
                right_curl = BicepCurl(right_shoulder, right_elbow, right_wrist, 
                                     ideal_angle=IDEAL_ANGLE, tolerance=TOLERANCE)
                
                # Evaluate both arms
                left_feedback, left_angle, left_percent = left_curl.evaluate()
                right_feedback, right_angle, right_percent = right_curl.evaluate()
                
                # Rep counting logic (left arm)
                if left_curl.is_arm_extended(threshold=EXTENDED_THRESHOLD):
                    if left_in_curl:
                        left_rep_count += 1
                        left_in_curl = False
                else:
                    if not left_in_curl and left_angle < (IDEAL_ANGLE + TOLERANCE):
                        left_in_curl = True
                
                # Rep counting logic (right arm)
                if right_curl.is_arm_extended(threshold=EXTENDED_THRESHOLD):
                    if right_in_curl:
                        right_rep_count += 1
                        right_in_curl = False
                else:
                    if not right_in_curl and right_angle < (IDEAL_ANGLE + TOLERANCE):
                        right_in_curl = True
                
                # Display only rep counts - clean and simple
                image = draw_text_with_sf_pro(image, f"LEFT: {left_rep_count}", 
                                            (10, 30), font_size=48, color=(255, 255, 0), shadow_color=(0, 0, 0), shadow_offset=(4, 4))
                image = draw_text_with_sf_pro(image, f"RIGHT: {right_rep_count}", 
                                            (10, 90), font_size=48, color=(255, 255, 0), shadow_color=(0, 0, 0), shadow_offset=(4, 4))
                
                # Draw key points for visualization
                cv2.circle(image, left_shoulder, 8, (255, 0, 0), -1)  # Blue for shoulder
                cv2.circle(image, left_elbow, 8, (0, 255, 0), -1)    # Green for elbow
                cv2.circle(image, left_wrist, 8, (0, 0, 255), -1)    # Red for wrist
                
                cv2.circle(image, right_shoulder, 8, (255, 0, 0), -1)  # Blue for shoulder
                cv2.circle(image, right_elbow, 8, (0, 255, 0), -1)    # Green for elbow
                cv2.circle(image, right_wrist, 8, (0, 0, 255), -1)   # Red for wrist
                
            except Exception as e:
                print(f"Error processing pose: {e}")
                image = draw_text_with_sf_pro(image, "Error processing pose", 
                                            (10, 30), font_size=24, color=(0, 0, 255), shadow_color=(0, 0, 0), shadow_offset=(3, 3))
        else:
            image = draw_text_with_sf_pro(image, 'No person detected', 
                                        (10, 30), font_size=28, color=(0, 0, 255), shadow_color=(0, 0, 0), shadow_offset=(3, 3))
        
        # Display the frame
        cv2.imshow('Simple Bicep Curl Demo', image)
        
        # Exit on pressing 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Demo finished!")
