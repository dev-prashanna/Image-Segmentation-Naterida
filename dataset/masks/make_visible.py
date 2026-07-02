import cv2
import numpy as np

mask = cv2.imread('frame_00002.png', cv2.IMREAD_GRAYSCALE)
visible = (mask * 50).astype(np.uint8)
cv2.imwrite('frame_00002_visible.png', visible)
print("Saved frame_00002_visible.png")
