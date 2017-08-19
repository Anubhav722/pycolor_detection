import cv2
import os
import numpy as np

def R1(r, g, b):
	if (r > 95) and (g > 40) and (b > 20) and ((max(r,max(g,b))) - min(r, min(g,b))) > 15 and (abs(r-g)>15) and (r>g) and (r>b):
		e1 = True
	else : 
		e1 = False
	if (r > 220) and (g > 210) and (b > 170) and (abs(r-g)<=15) and (r>b) and (g>b):
		e2 = True
	else :
		e2 = False
	
	return (e1 or e2)

def R2(Y, Cr, Cb):
	if Cr <= (1.5862*Cb+20):
		e3 = True
	else :
		e3 = False
	if Cr >= (0.3448*Cb+76.2069):
		e4 = True
	else :
		e4 = False
	if Cr >= (-4.5652*Cb+234.5652):
		e5 = True
	else :
		e5 = False
	if Cr <= (-1.15*Cb+301.75):
		e6 = True
	else:
		e6 = False
	if Cr <= (-2.2857*Cb+432.85):
		e7 = True
	else:
		e7 = False

	return (e3 and e4 and e5 and e6 and e7)

def R3(H, S, V):
	return ((H<25) or (H > 230))

def get_skin(img):
	dst = img.copy()
	cwhite = 255
	cblack = 0

	img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
	img_hsv = np.float32(img)
	img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	img_hsv = cv2.normalize(img_hsv,img_hsv, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC3)


	for i in xrange(img.shape[0]):
		for j in xrange(img.shape[1]):
			pix_bgr = img[i][j]
			b = pix_bgr[0]
			g = pix_bgr[1]
			r = pix_bgr[2]

			a = R1(r, g, b)

			pix_ycrcb = img_ycrcb[i][j]
			Y = pix_ycrcb[0]
			Cr = pix_ycrcb[1]
			Cb = pix_ycrcb[2]

			b = R2(Y, Cr, Cb)

			pix_hsv = img_hsv[i][j]
			H = pix_hsv[0]
			S = pix_hsv[1]
			V = pix_hsv[2]

			c = R3(H, S, V)

			if (a and b and c):
				dst[i][j] = cblack

	return dst





def skin_remove(img):
	skin = get_skin(img);
	return skin
	
