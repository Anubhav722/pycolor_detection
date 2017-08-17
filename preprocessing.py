#Parth Sharma#
import cv2
import os
import pandas as pd
import matplotlib.colors as colors
from colormap import rgb2hex
import numpy as np
import time
import sys
from math import sqrt
import random
from collections import defaultdict





def resizing(img):
	height, width, channels = img.shape
	if max(height,width) > 500:
		ratio = float(height)/width
		new_width = 500/ratio    
		img_resized = cv2.resize(img,(int(new_width),500))
		ip_convert = cv2.imencode('.png',img_resized)
	else:
		ip_convert = cv2.imencode('.png',img)

	return ip_convert
def Sobel (channel):
	sobelx = cv2.Sobel(channel, cv2.CV_16S, 1, 0, ksize =3)
	sobely = cv2.Sobel(channel, cv2.CV_16S, 0, 1, ksize =3)
	sobel = np.hypot(sobelx, sobely)
	sobel[sobel > 255] = 255

	return sobel

def findSignificantContours (img, sobel_8u,sobel):


	image, contours, heirarchy = cv2.findContours(sobel_8u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# Find level 1 contoursimge.shape[:2], dtype="uint8"
	mask = np.ones(image.shape[:2], dtype="uint8") * 255

	level1 = []
	for i, tupl in enumerate(heirarchy[0]):
		# Each array is in format (Next, Prev, First child, Parent)
		# Filter the ones without parent
		if tupl[3] == -1:
			tupl = np.insert(tupl, 0, [i])
			level1.append(tupl)
	# From among them, find the contours with large surface area.
	significant = []
	tooSmall = sobel_8u.size * 5 / 100 # If contour isn't covering 5% of total area of image then it probably is too small
	for tupl in level1:
		contour = contours[tupl[0]];
		area = cv2.contourArea(contour)
		if area > tooSmall:
			cv2.drawContours(mask, [contour],0, (0,255,0),2, cv2.LINE_AA, maxLevel=1)
			significant.append([contour, area])
	significant.sort(key=lambda x: x[1])
	significant =  [x[0] for x in significant];
	peri = cv2.arcLength(contour, True)
	approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
	mask = sobel.copy()
	mask[mask > 0] = 0
	cv2.fillPoly(mask, significant, 255,0)
	mask = np.logical_not(mask)
  	img[mask] = 0;

	return img

def image_segmentation(ip_convert):
	img = cv2.imdecode(np.squeeze(np.asarray(ip_convert[1])),1)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (3,3), 0) 
	sobel = Sobel(blurred)
	mean = np.mean(sobel);
	# Zero any value that is less than mean.
	# This reduces a lot of noise. 
	sobel[sobel <= mean] = 0;
	sobel[sobel > 255] = 255;
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
	sobel = cv2.morphologyEx(sobel, cv2.MORPH_CLOSE, kernel,iterations = 2)
	sobel_8u = np.asarray(sobel, np.uint8)
	significant = findSignificantContours(img, sobel_8u.copy(),sobel)
	segmented_img = cv2.imencode('.png', significant)

	return segmented_img

def removebg(segmented_img):
	
	src = cv2.imdecode(np.squeeze(np.asarray(segmented_img[1])), 1)
	tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
	_,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
	b, g, r = cv2.split(src)
	rgba = [b,g,r, alpha]
	dst = cv2.merge(rgba,4)
	processed_img = cv2.imencode('.png', dst)

	return processed_img




