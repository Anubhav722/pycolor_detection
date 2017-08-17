import pycolor
import preprocessing
import os
import sys
import cv2
import csv

if __name__ == '__main__':
	try :
		images = os.listdir(sys.argv[1])

	except Exception as e:
		print ("path not a directory path...")
		print ("Loading image path...")
		try :
			image = cv2.imread(sys.argv[1])
		except:
			print ("path not found!!!")

	if images:
		for image in images:
			try:
				img  = cv2.imread(sys.argv[1]+image)
			except:
				img = cv2.imread(sys.argv[1] + "/" + image)
			ip_converted = preprocessing.resizing(img)
			segmented_image = preprocessing.image_segmentation(ip_converted)
			processed_image = preprocessing.removebg(segmented_image)

			detect = pycolor.dcolor(processed_image,sys.argv[2])
			print image,detect
			# map_data = color_detect.data()
			with open(sys.argv[3],'a+') as result:
				writer = csv.writer(result,delimiter=',')
				writer.writerow([image,str(detect)])





	
