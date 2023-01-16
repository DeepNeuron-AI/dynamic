import cv2
import numpy as np

from pathlib import Path
import sys

IMAGE_ONLY_WINDOW = "Image only"
SCRIBBLE_ONLY_WINDOW = "Scribble only"
IMAGE_AND_SCRIBBLE_WINDOW = "Image + Scribble"
DENOISED_WINDOW = "Denoised"
BLURRED_WINDOW = "Blurred"

# Drawing parameters
LINE_THICKNESS = 3

IMAGE_FP = Path("images/my_image.png")
OUTPUT_FP = Path("images/my_image_scribble.png")


image_only = cv2.imread(str(IMAGE_FP))

# scribble_only should be grayscale
scribble_only = np.ones(image_only.shape[:-1], dtype=image_only.dtype) * 255
image_and_scribble = image_only.copy()

denoised = cv2.fastNlMeansDenoisingColored(image_only, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
blurred = cv2.GaussianBlur(image_only, (5, 5), 0)

cv2.namedWindow(SCRIBBLE_ONLY_WINDOW, cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow(IMAGE_ONLY_WINDOW, cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow(IMAGE_AND_SCRIBBLE_WINDOW, cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow(DENOISED_WINDOW, cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow(BLURRED_WINDOW, cv2.WINDOW_GUI_NORMAL)

cv2.imshow(IMAGE_ONLY_WINDOW, image_only)
cv2.imshow(SCRIBBLE_ONLY_WINDOW, scribble_only)
cv2.imshow(IMAGE_AND_SCRIBBLE_WINDOW, scribble_only)
cv2.imshow(DENOISED_WINDOW, denoised)
cv2.imshow(BLURRED_WINDOW, blurred)

cv2.resizeWindow(IMAGE_ONLY_WINDOW, 512, 512)
cv2.resizeWindow(SCRIBBLE_ONLY_WINDOW, 512, 512)
cv2.resizeWindow(IMAGE_AND_SCRIBBLE_WINDOW, 512, 512)
cv2.resizeWindow(DENOISED_WINDOW, 512, 512)
cv2.resizeWindow(BLURRED_WINDOW, 512, 512)

# # This just doesn't work lol
# image_x, image_y, width, height = cv2.getWindowImageRect(IMAGE_WINDOW)
# cv2.moveWindow(SCRIBBLE_WINDOW, int(image_x + 1.05*width), image_y)



drawing = False
erasing = False


def get_slice(x: int, y: int, width: int, height: int, thickness: int):
    min_y = y-thickness//2
    max_y = y+thickness//2
    min_x = x-thickness//2
    max_x = x+thickness//2
    
    min_y, max_y = np.clip([min_y, max_y], 0, height-1)
    min_x, max_x = np.clip([min_x, max_x], 0, width-1)
    
    min_y = int(min_y)
    max_y = int(max_y)
    min_x = int(min_x)
    max_x = int(max_x)

    return min_y, max_y, min_x, max_x



def draw_callback(event: int, x: int, y: int, flags: int, *userdata):
    global drawing, erasing, scribble_only, image_and_scribble
    height, width = scribble_only.shape

    def draw():
        global scribble_only, image_and_scribble
        min_y, max_y, min_x, max_x = get_slice(x,y,width, height, LINE_THICKNESS)
        print(min_y, max_y, min_x, max_x)
        scribble_only[min_y:max_y,min_x:max_x] = 0
        image_and_scribble[min_y:max_y,min_x:max_x] = [0, 0, 255]

    def erase():
        global scribble_only, image_and_scribble
        min_y, max_y, min_x, max_x = get_slice(x,y,width, height, LINE_THICKNESS*1.5)
        print(min_y, max_y, min_x, max_x)
        scribble_only[min_y:max_y,min_x:max_x] = 255
        image_and_scribble[min_y:max_y,min_x:max_x] = image_only[min_y:max_y,min_x:max_x]

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        erasing = False
        print(f"Clicked Left Mouse @ ({x}, {y})")
        draw()
    elif event == cv2.EVENT_RBUTTONDOWN:
        erasing = True
        drawing = False
        print(f"Clicked Right Mouse @ ({x}, {y})")
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        print(f"Released Left Mouse @ ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONUP:
        erasing = False
        print(f"Released Right Mouse @ ({x}, {y})")
    elif event == cv2.EVENT_MOUSEMOVE:
        print(f"Moved Mouse @ ({x}, {y})")
        if drawing:
            draw()
        if erasing:
            erase()
    


cv2.setMouseCallback(IMAGE_AND_SCRIBBLE_WINDOW, draw_callback)
cv2.setMouseCallback(IMAGE_ONLY_WINDOW, draw_callback)


try:
    while True:
        cv2.imshow(IMAGE_ONLY_WINDOW, image_only)
        cv2.imshow(SCRIBBLE_ONLY_WINDOW, scribble_only)
        cv2.imshow(IMAGE_AND_SCRIBBLE_WINDOW, image_and_scribble)

        # Press Q on keyboard to  exit
        keypress = cv2.waitKey(25) & 0xFF
        if keypress == ord('q'):
            break


    if input(f"Do you want to save that scribble to {OUTPUT_FP}?\n") in ("y", "Y"):
        print("Saving image")
        # scribble_only = cv2.cvtColor()
        cv2.imwrite(str(OUTPUT_FP), scribble_only)
    else:
        print("Quitting without saving")
finally:
    cv2.destroyAllWindows()