import cv2
import numpy as np

# Create a black image
img = np.zeros((512, 512, 3), np.uint8)

# Draw a white rectangle
cv2.rectangle(img, (100, 100), (400, 400), (255, 255, 255), 2)

# Show the image
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


