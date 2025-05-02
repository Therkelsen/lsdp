import cv2
import os

# --- Config ---
video_path = "Data for miniproject on visual odometry/DJI_0199.MOV"
output_folder = "saved_frames"
frame_interval = 25
skip_initial_frames = 1200

# --- Setup Output Folder ---
os.makedirs(output_folder, exist_ok=True)

# --- Open Video ---
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    frame_count += 1

    # Skip the first N frames
    if frame_count <= skip_initial_frames:
        continue

    # Save every Nth frame
    if (frame_count - skip_initial_frames) % frame_interval == 0:
        filename = f"frame_{saved_count:04d}.jpg"
        filepath = os.path.join(output_folder, filename)
        cv2.imwrite(filepath, frame)
        saved_count += 1

cap.release()
print(f"Saved {saved_count} frames to '{output_folder}'.")
