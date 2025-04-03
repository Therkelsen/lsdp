import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from lsdp_tools import FrameIterator


def draw_points(frame, points, color):
    """Draw points on the frame."""
    for point in points:
        cv.circle(frame, tuple(point), 5, color, 3)


def add_frame_counter(frame, counter, org, font, font_scale, color, thickness):
    """Add a frame counter as text to the frame."""
    return cv.putText(frame, "%d" % counter, org, font, font_scale, color, thickness, cv.LINE_AA)


def extract_colors(frame, points):
    """Extract RGB color values for the given points in the frame."""
    return [frame[point[1], point[0]].tolist() for point in points]


def motion_detector_acc(frame, accumulator_image, frame_count):
    """Detect motion by comparing the current frame with the average of earlier frames."""
    if accumulator_image is None:
        accumulator_image = np.zeros_like(frame, dtype=np.float32)

    # Add the current frame to the accumulator
    accumulator_image += frame.astype(np.float32)

    # Ensure frame_count is at least 1 to avoid division by zero
    frame_count = max(frame_count, 1)

    # Calculate the mean image (divide by frame_count to average)
    mean_image = accumulator_image / frame_count

    # Convert mean image to uint8 for visualization
    mean_image_uint8 = np.clip(mean_image, 0, 255).astype(np.uint8)

    # Calculate absolute difference
    diff = cv.absdiff(frame, mean_image_uint8)
    diff = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)

    return mean_image_uint8, diff, accumulator_image


def motion_detector_ewma(frame, ewma_image, alpha=0.05):
    """Detect motion using an Exponentially Weighted Moving Average (EWMA)."""
    if ewma_image is None:
        ewma_image = frame.astype(np.float32)  # Initialize with first frame

    # Update EWMA using the current frame
    ewma_image = alpha * frame.astype(np.float32) + (1 - alpha) * ewma_image

    # Convert EWMA image to uint8 for visualization
    ewma_image_uint8 = np.clip(ewma_image, 0, 255).astype(np.uint8)

    # Calculate absolute difference
    diff = cv.absdiff(frame, ewma_image_uint8)
    diff = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)

    return ewma_image_uint8, diff, ewma_image



def visualize_color_changes(bg_colors_over_time, fg_colors_over_time):
    """Visualize the RGB color changes over time for each pixel."""
    bg_colors_over_time = np.array(bg_colors_over_time)
    fg_colors_over_time = np.array(fg_colors_over_time)

    num_bg_points = bg_colors_over_time.shape[1]
    num_fg_points = fg_colors_over_time.shape[1]

    plt.figure(figsize=(15, 10))

    # Plot background colors for each point
    for i in range(num_bg_points):
        plt.subplot(2, num_bg_points, i + 1)
        plt.title(f"BG Point {i + 1}")
        for j, (color, color_name) in enumerate(zip(["red", "green", "blue"], ["R", "G", "B"])):
            plt.plot(bg_colors_over_time[:, i, j], label=f"{color_name}", color=color)
        plt.ylim(0, 255)
        plt.xlabel("Frame")
        plt.ylabel("Color Intensity")
        plt.legend()

    # Plot foreground colors for each point
    for i in range(num_fg_points):
        plt.subplot(2, num_fg_points, num_bg_points + i + 1)
        plt.title(f"FG Point {i + 1}")
        for j, (color, color_name) in enumerate(zip(["red", "green", "blue"], ["R", "G", "B"])):
            plt.plot(fg_colors_over_time[:, i, j], label=f"{color_name}", color=color)
        plt.ylim(0, 255)
        plt.xlabel("Frame")
        plt.ylabel("Color Intensity")
        plt.legend()

    plt.tight_layout()
    plt.show()


def process_frames(video_path, bg_points, fg_points, output_path, show_video=True):
    """Process video frames, draw points, extract colors, and handle user input."""
    fi = FrameIterator(video_path)
    generator = fi.frame_generator()
    total_frames = int(cv.VideoCapture(video_path).get(cv.CAP_PROP_FRAME_COUNT))
    
    accumulator_image = None
    ewma_image = None

    # Font settings
    font = cv.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    font_scale = 1
    color = (255, 0, 0)
    thickness = 2

    counter = 0
    bg_colors_over_time = []
    fg_colors_over_time = []

    for frame in generator:
        counter += 1

        # Extract colors for background and foreground points
        bg_colors = extract_colors(frame, bg_points)
        fg_colors = extract_colors(frame, fg_points)
        bg_colors_over_time.append(bg_colors)
        fg_colors_over_time.append(fg_colors)

        mean_acc, diff_acc, accumulator_image = motion_detector_acc(frame, accumulator_image, counter)
        mean_ewma, diff_ewma, ewma_image = motion_detector_ewma(frame, ewma_image, alpha=0.05)

        # Plot results only for the last frame
        if counter == total_frames:
            plt.figure(figsize=(15, 10))

            plt.subplot(2, 3, 1)
            plt.imshow(cv.cvtColor(accumulator_image.astype(np.uint8), cv.COLOR_BGR2RGB))
            plt.title("Accumulator Image")

            plt.subplot(2, 3, 2)
            plt.imshow(cv.cvtColor(mean_acc, cv.COLOR_BGR2RGB))
            plt.title("Mean Image (Acc)")

            plt.subplot(2, 3, 3)
            plt.imshow(diff_acc, cmap='gray')
            plt.title("Diff Image (Acc)")

            plt.subplot(2, 3, 4)
            plt.imshow(cv.cvtColor(ewma_image.astype(np.uint8), cv.COLOR_BGR2RGB))
            plt.title("EWMA Image")

            plt.subplot(2, 3, 5)
            plt.imshow(cv.cvtColor(mean_ewma, cv.COLOR_BGR2RGB))
            plt.title("Mean Image (EWMA)")

            plt.subplot(2, 3, 6)
            plt.imshow(diff_ewma, cmap='gray')
            plt.title("Diff Image (EWMA)")

            plt.tight_layout()
            plt.show()

        if show_video:
            # Draw background and foreground points
            draw_points(frame, bg_points, (0, 255, 255))
            draw_points(frame, fg_points, (255, 255, 0))

            # Add frame counter
            frame = add_frame_counter(frame, counter, org, font, font_scale, color, thickness)

            # Display the frame
            cv.imshow('frame', frame)
            k = cv.waitKey(30) & 0xff
            if k == 27:  # Exit on 'ESC'
                break
            elif k == ord('s'):  # Save frame on 's'
                cv.imwrite(output_path, frame)
        else:
            print(f"Processing frame {counter+1} / {total_frames}")

    if show_video:
        cv.destroyAllWindows()

    # Visualize color changes over time
    visualize_color_changes(bg_colors_over_time, fg_colors_over_time)


if __name__ == "__main__":
    video_path = 'lecture_3/03_image_sequences/input/Sometimes Security Cameras catch a gem.mp4'
    bg_points = np.array([[273, 395], [600, 311]])
    fg_points = np.array([[156, 102], [388, 248]])
    output_path = "../output/ex01stillimage.png"

    # Set show_video to False to process in the background
    process_frames(video_path, bg_points, fg_points, output_path, show_video=False)
