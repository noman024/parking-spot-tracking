import cv2
import os
import torch
from ultralytics import YOLO
import json
from centroid_tracker import CentroidTracker
import random
import string

# Create an instance of the CentroidTracker with max_distance parameter
ct = CentroidTracker(max_disappeared=50, max_distance=50)

def extract_frames(video_path, output_dir, detection_dir, entry_area, duration=25):
    """
    Extract frames from a video and save them as images.

    Parameters:
    - video_path: str, the path to the video file
    - output_dir: str, the directory to save the extracted frames
    - detection_dir: str, the directory to save detection results
    - entry_area: tuple, defining the area for object detection
    - duration: int, the maximum duration in seconds to extract frames (default is 25)

    Returns:
    None
    """
    # Create output directories if they don't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(detection_dir):
        os.makedirs(detection_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= duration * fps:
            break

        frame_count += 1

        # Save original frame
        frame_filename = f"frame_{frame_count}.jpg"  # Extract just the filename
        frame_path = os.path.join(output_dir, frame_filename)  # Construct full path
        cv2.imwrite(frame_path, frame)
        print(f"Frame {frame_count} saved.")
        
        entry_area = (0.6044635416666666, 0.3456018518518518, 0.15838020833333333, 0.12032407407407408)
        entry_area = [entry_area[0] * frame.shape[1], entry_area[1] * frame.shape[0], entry_area[2] * frame.shape[1], entry_area[3] * frame.shape[0]]
        # Object detection (YOLOv9)
        detect_objects(frame, detection_dir, frame_filename, entry_area)

    cap.release()

def detect_objects(frame, detection_dir, output_filename, entry_area):
    """
    A function to detect objects in a frame using a YOLOv9c model and centroid tracking.
    
    Parameters:
    - frame: The input frame to detect objects in.
    - detection_dir: The directory to save detection results.
    - output_filename: The filename for the output frame with detections.
    - entry_area: The area in the frame where objects are expected to enter.
    
    Returns:
    None
    """
    # Check if CUDA (GPU support) is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build a YOLOv9c model from pretrained weight
    model = YOLO('/home/noman/parking-spot-tracking/yolov9c.pt').to(device)
        
    # Detect objects in frame
    results = model(frame)
    
    # Initialize a list to store detected objects
    detected_objects = []
    
    # Extract centroids from detected objects
    centroids = []
    
    # Loop through all images in the batch
    for _, result in enumerate(results):
        boxes = result.boxes.xyxy.cpu().numpy()
        
        # Loop through detected objects in the current image
        for box in boxes:
            # Get centroid of the object (bounding box center)
            centroid_x = (box[0] + box[2]) / 2
            centroid_y = (box[1] + box[3]) / 2
            centroids.append((centroid_x, centroid_y))

    # Update centroid tracker with detected centroids
    ct.update(centroids)
    
    # Retrieve objects from centroid tracker
    objects = ct.objects
    
    # Loop through objects and add to detected_objects list
    for object_id, centroid in objects.items():
        # Construct dictionary for the detected object
        unique_id = generate_unique_id()  # Generate unique ID for the vehicle
        obj = {
            "id": unique_id,
            "object_id": object_id,
            "filename": output_filename,
            "centroid": centroid,
            "entry_area": entry_area
        }
        detected_objects.append(obj)

    # Visualize and save the frame with detections if there are any detected objects
    if len(detected_objects) > 0:
        print(f"Detected {len(detected_objects)} objects in frame {output_filename}.")
        # Plot only the detected objects within the entry area
        for obj in detected_objects:
            # Draw centroid
            centroid_x, centroid_y = obj['centroid']
            cv2.circle(frame, (int(centroid_x), int(centroid_y)), 4, (0, 255, 0), -1)
            # Write unique ID above the centroid
            cv2.putText(frame, str(obj['object_id']), (int(centroid_x) - 10, int(centroid_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Save the frame with detections in the detection directory
        detection_path = os.path.join(detection_dir, output_filename)  # Construct full output path
        cv2.imwrite(detection_path, frame)
        
        # Save detected objects to JSON file
        json_output_path = os.path.join(detection_dir, output_filename.replace('.jpg', '.json'))
        with open(json_output_path, 'w') as json_file:
            json.dump(detected_objects, json_file)

def generate_unique_id():
    """
    Generate a random alphanumeric ID in the format "XY-1234"
    """
    # Generate a random alphanumeric ID in the format "XY-1234"
    prefix = ''.join(random.choices(string.ascii_uppercase, k=2))
    suffix = ''.join(random.choices(string.digits, k=4))
    return f"{prefix}-{suffix}"


if __name__ == "__main__":
    video_path = "/home/noman/parking-spot-tracking/Parking-Spot-Tracking/data/overhead_cctv_video.mp4"  # Path to overhead camera video
    output_dir = "/home/noman/parking-spot-tracking/frames"  # Directory to save extracted frames
    detection_dir = "/home/noman/parking-spot-tracking/detections"  # Directory to save frames with detections
    duration = 25  # Duration in seconds

    # Entry area coordinates (xmin, ymin, xmax, ymax)
    entry_area = (0.6044635416666666, 0.3456018518518518, 0.15838020833333333, 0.12032407407407408)

    extract_frames(video_path, output_dir, detection_dir, entry_area, duration)

    print("Frames extracted and objects detected within the entry area successfully.")
