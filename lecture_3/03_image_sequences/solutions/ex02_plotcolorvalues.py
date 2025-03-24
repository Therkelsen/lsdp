import numpy as np
from lsdp_tools import FrameIterator
import matplotlib.pyplot as plt


def main():
    fi = FrameIterator('../input/remember.mkv')
    generator = fi.frame_generator()

    # Define locations from which to extract pixel information
    foreground = np.array([[177, 73], [256, 146]])
    background = np.array([[456, 378], [581, 306], [125, 278]])

    values = np.array([])
    foreground_pixel_values = None
    background_pixel_values = None

    for frame in generator:

        # Collect pixel values from background locations
        pixels = np.array([])
        for location in background:
            temp_value = frame[location[1], location[0], :]
            pixels = np.append(pixels, temp_value)

        if background_pixel_values is not None:
            background_pixel_values = np.concatenate((background_pixel_values,
                    [pixels]))
        else:
            background_pixel_values = [pixels]

        # Collect pixel values from foreground locations
        pixels = np.array([])
        for location in foreground:
            temp_value = frame[location[1], location[0], :]
            pixels = np.append(pixels, temp_value)

        if foreground_pixel_values is not None:
            foreground_pixel_values = np.concatenate((foreground_pixel_values,
                    [pixels]))
        else:
            foreground_pixel_values = [pixels]



    # Visualize pixel values over frames.
    ax = plt.subplot(2, 1, 1)
    for k in range(3*foreground.shape[0]):
        plt.plot(foreground_pixel_values[:, k])
    ax.set_title("Foreground")

    ax = plt.subplot(2, 1, 2)
    for k in range(3*background.shape[0]):
        plt.plot(background_pixel_values[:, k])
    ax.set_title("Background")

    plt.show()

    
main()

