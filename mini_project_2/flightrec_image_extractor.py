import csv
import cv2
import numpy as np
import os

############### Exercise 9.1.2 ###############

def remove_non_empty_dir(path):
    if os.path.exists(path):
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(path)
        print(f"Deleted folder and all contents: {path}")
    else:
        print(f"Folder does not exist: {path}")
        

def extract_video_log_sequences(video_path, csv_path, fps=25):
    import numpy as np
    import cv2

    # Load CSV with headers
    data = np.genfromtxt(csv_path, delimiter=',', dtype=None, names=True, encoding='utf-8')

    required_fields = ['CUSTOMisVideo', 'CUSTOMupdateTime', 'OSDlatitude', 'OSDlongitude', 'OSDaltitude_m']
    for field in required_fields:
        if field not in data.dtype.names:
            raise ValueError(f"Missing required column: {field}")

    is_video = data['CUSTOMisVideo']
    frame_interval_ms = 1000 / fps

    # Identify start and end indices of each recording sequence
    sequences_indices = []
    in_sequence = False
    start_idx = None

    for i, val in enumerate(is_video):
        if not in_sequence and val == "Recording":
            start_idx = i
            in_sequence = True
        elif in_sequence and (val == "Stop" or (i == len(is_video) - 1 and val == "Recording")):
            end_idx = i
            sequences_indices.append((start_idx, end_idx))
            in_sequence = False

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video file.")

    all_sequences = []

    for seq_num, (start_idx, end_idx) in enumerate(sequences_indices):
        print(f"Processing sequence {seq_num+1}: rows {start_idx} to {end_idx}")
        result = []

        for i in range(start_idx, end_idx + 1, 25):  # sample every 25 rows
            video_timestamp = i * frame_interval_ms  # <-- absolute time
            cap.set(cv2.CAP_PROP_POS_MSEC, video_timestamp)
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame at {video_timestamp:.2f} ms (row {i})")
                break

            update_time = data['CUSTOMupdateTime'][i]
            latitude = data['OSDlatitude'][i]
            longitude = data['OSDlongitude'][i]
            altitude = data['OSDaltitude_m'][i]

            result.append((i, frame, update_time, (latitude, longitude, altitude)))

        all_sequences.append(result)

    cap.release()
    return all_sequences


# --- Config ---
video_path = "mini_project_2/Data for miniproject on visual odometry/DJI_0199.MOV"
log_path = "mini_project_2/Data for miniproject on visual odometry/DJIFlightRecord_2021-03-18_[13-04-51]-TxtLogToCsv.csv"
output_folder = "mini_project_2/saved_sequences"
# output_folder = "mini_project_2/saved_frames"

sequences = extract_video_log_sequences(video_path, log_path, fps=25)

print(f"\nFound {len(sequences)} sequences.")

for idx, seq in enumerate(sequences):
    # Step 1: Create folder for sequence
    sequence_path = os.path.join(output_folder, f"sequence_{idx+1}")
    print(f"Creating dir for Sequence {idx+1} at \"{sequence_path}\"")
    remove_non_empty_dir(sequence_path)
    os.makedirs(sequence_path, exist_ok=True)

    # Step 2: Debug info
    print(f"Sequence {idx+1} has {len(seq)} frames.")
    if seq:
        _, frame, time, attitude = seq[0]
        print(f"First frame of Sequence {idx+1}:")
        print(f"Time: {time}")
        print(f"Attitude: {attitude}")
        print(f"Frame shape: {frame.shape}\n")

    # Step 3: Save all frames + write metadata to CSV
    csv_path = os.path.join(sequence_path, "frames.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "time", "latitude", "longitude", "altitude", "sequence_id"])  # header

        for frame_idx, (global_idx, frame, time, attitude) in enumerate(seq):
            filename = f"frame_{global_idx:06d}.jpg"
            filepath = os.path.join(sequence_path, filename)
            cv2.imwrite(filepath, frame)

            lat, lon, alt = attitude
            writer.writerow([filename, time, lat, lon, alt, idx+1])