#
#
# credit: David G. Lowe, SIFT algorithm (https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
# credit: Sergio Canu (Pysource), source code (https://pysource.com/2018/03/21/feature-detection-sift-surf-obr-opencv-3-4-with-python-3-tutorial-25/)

import cv2
import numpy as np
import math
import sys

img = cv2.imread("grandcanyon_jimmkidd-alamy.jpg", cv2.IMREAD_GRAYSCALE)

sift = cv2.xfeatures2d.SIFT_create()
kp_image, desc_image = sift.detectAndCompute(img, None)

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

cap = cv2.VideoCapture(0)

def getHomography(frameGray):
	kp_grayframe, desc_grayframe = sift.detectAndCompute(frameGray, None)

	if len(kp_grayframe) > 3:
		matches = flann.knnMatch(desc_image, desc_grayframe, k=2)

		good_points = []
		for m, n in matches:
			if m.distance < 0.5*n.distance: 
				good_points.append(m)

		if len(good_points) > 8:
			query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
			train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

			matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
			matches_mask = mask.ravel().tolist()

			if matrix is not None:
				h, w = img.shape
				pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
				dst = cv2.perspectiveTransform(pts, matrix)
				homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
				
				cv2.imshow("Homography", homography)
			else:
				cv2.imshow("Homography", frameGray)

		else:
			cv2.imshow("Homography", frameGray)
	else:
		cv2.imshow("Homography", frameGray)

while(True):
	success, frame = cap.read();

	frameBlur = cv2.GaussianBlur(frame, (7, 7), 1)
	frameGray = cv2.cvtColor(frameBlur, cv2.COLOR_BGR2GRAY)

	getHomography(frameGray)
	
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()