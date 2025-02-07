import cv2
import matplotlib
import matplotlib.pyplot as plt

# Load image and extract values from it
selected_line = 100
img = cv2.imread("../input/flower.jpg")
assert img is not None, "Failed to load image."

green_values = img[selected_line, :, 1]
blue_values = img[selected_line, :, 2]
red_values = img[selected_line, :, 0]

# Plot green values and save the plot to file
plt.plot(green_values)
plt.savefig("../output/ex08_plot.png")

# Plot red, green and blue values and save the plot to file
plt.plot(green_values, 'g')
plt.plot(blue_values, 'b')
plt.plot(red_values, 'r')
plt.savefig("../output/ex08_plot_rgb.png")

# Mark the selected line in the image and save it to file
img[selected_line, :, :] = 255
cv2.imwrite("../output/ex08_image.jpg", img)

