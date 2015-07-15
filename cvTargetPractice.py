import argparse
import cv2

# Imports for frame analysis
import imutils
import numpy as np

# Imports for raspberry pi
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

# ==================================================================
def analyzeOneFrame(aFrame):
	# TODO: resize frame
	status = "No targets."
	image = aFrame
	processed = image.copy()
	shapeMask = cv2.inRange(image, np.array([0,0,100]), np.array([110,110,255]))
	cv2.imshow("shapeMask", shapeMask)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(shapeMask, (7,7), 0)
	autoEdged = imutils.autoCanny(blurred)

	(cnts, _) = cv2.findContours(autoEdged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.01 * peri, True)

		cv2.drawContours(processed, [c], -1, (255,0,0), 2)

		if len(approx) >= 4 and len(approx) <= 6:
			(x,y,w,h) = cv2.boundingRect(approx)
			aspectRatio = w / float(h)
			area = cv2.contourArea(c)
			hullArea = cv2.contourArea(cv2.convexHull(c))
			solidity = area / float(hullArea)

			keepDims = w > 25 and h > 25
			keepSolidity = solidity > 0.9
			keepAspectRatio = aspectRatio >= 0.8 and aspectRatio <= 1.2

			conditionsString = "dims: " + str(keepDims) + ", solidity: " + str(keepSolidity) + ", aspect: " + str(keepAspectRatio)
			cv2.putText(processed, conditionsString, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
			if keepDims and keepSolidity and keepAspectRatio:
				cv2.drawContours(processed, [approx], -1, (0,0,255), 4)
				status = "Target(s) acquired."

				M = cv2.moments(approx)
				(cX,cY) = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
				(startX, endX) = (int(cX - (w * 0.15)), int(cX + (w * 0/15)))
				(startY, endY) = (int(cY - (h * 0.15)), int(cY + (h * 0.15)))
				cv2.line(processed, (startX, cY), (endX, cY), (0,0,255), 3)
				cv2.line(processed, (cX, startY), (cX, endY), (0,0,255), 3)

	cv2.putText(processed, status, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

	cv2.imshow("Raw", image)
	cv2.imshow("Processed", processed)
	#cv2.imshow("Intermediates", np.hstack([blurred,autoEdged]))
# ==================================================================
def loopPiCamera():
	# initialize the camera and grab a reference to the raw camera capture
	camera = PiCamera()
	camera.resolution = (640, 480)
	camera.framerate = 32
	rawCapture = PiRGBArray(camera, size=(640, 480))

	# allow the camera to warmup
	time.sleep(0.1)

	# capture frames from the camera
	for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
		analyzeOneFrame(frame.array)

		key = cv2.waitKey(1) & 0xFF
		# clear the stream in preparation for the next frame
		rawCapture.truncate(0)
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

def loopVideo(videoFile):
	camera = cv2.VideoCapture(videoFile)
	while True:
		(grabbed, frame) = camera.read()
		if not grabbed:
			break

		analyzeOneFrame(frame)

		key = cv2.waitKey(1) & 0xFF
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

def loopWebcam():
	camera = cv2.VideoCapture(0)
	while True:
		(grabbed, frame) = camera.read()
		
		analyzeOneFrame(frame)

		key = cv2.waitKey(1) & 0xFF
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

def analyzeImage(imageFile):
	image = cv2.imread(imageFile)
	analyzeOneFrame(image)
	cv2.waitKey(0)

# ==================================================================
def generalCleanUp():
	cv2.destroyAllWindows()

# ==================================================================

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help = "path to the (optional) video file")
ap.add_argument("-i", "--image", help = "path to the (optional) image file")
ap.add_argument("-r", "--raspberry", help = "flag for using the raspberry pi camera")
ap.add_argument("-w", "--webcam", help = "flag for using the webcam")

args = vars(ap.parse_args())

if args.get("raspberry", False):
	loopPiCamera()
elif args.get("video", False):
	loopVideo(args["video"])
elif args.get("image", False):
	analyzeImage(args["image"])
elif args.get("webcam", False):
	loopWebcam()

generalCleanUp()
