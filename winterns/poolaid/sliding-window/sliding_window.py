# USAGE
# python sliding_window.py --image images/adrian_florida.jpg

from pyimagesearch.helpers import pyramid
from pyimagesearch.helpers import sliding_window
import argparse
import time
import cv2

print(cv2.__version__)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
(winW, winH) = (128, 128)

for resized in pyramid(image, scale=1.5):
	for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
		if window.shape[0] != winH or window.shape[1] != winW:
			continue
			
		# MACHINE LEARNING CLASSIFIER WITH WINDOW

		clone = resized.copy()
		cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
		cv2.imshow("Window", clone)
		cv2.waitKey(1)
		time.sleep(0.025)
