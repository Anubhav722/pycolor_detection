import pycolor
import preprocessing
import os
import sys
import cv2
import csv

if __name__ == '__main__':
    images = list()
    try :
        images = os.listdir(sys.argv[1])

    except Exception as e:
        print ("path not a directory path...")
        print ("Loading image path...")
        try :
            image = cv2.imread(sys.argv[1])
        except:
            print ("path not found!!!")

    if images != []:
        try :
            output_csv = sys.argv[3]
        except :
            output_csv = "result.csv"
        with open(output_csv,'wb+') as result:
            writer = csv.writer(result,delimiter=',')
            writer.writerow(['Image','Predictions'])
            for image in images:
                try:
                    img  = cv2.imread(sys.argv[1]+image)
                except:
                    img = cv2.imread(sys.argv[1] + "/" + image)
                print image
                ip_converted = preprocessing.resizing(img)
                segmented_image = preprocessing.image_segmentation(ip_converted)
                processed_image = preprocessing.removebg(segmented_image)
                try :
                    detect = pycolor.detect_color(processed_image,sys.argv[2])
                except:
                    detect = pycolor.detect_color(processed_image,"colors.csv")
                print (image,detect)
                writer.writerow([image,str(detect)])
    else :
        try :
            output_csv = sys.argv[3]
        except :
            output_csv = "result.csv"
        with open(output_csv,'wb+') as result:
            writer = csv.writer(result,delimiter=',')
            writer.writerow(['Image','Predictions'])
            
            try:
                img  = cv2.imread(sys.argv[1])
            except:
                img = cv2.imread(sys.argv[1])
            
            ip_converted = preprocessing.resizing(img)
            segmented_image = preprocessing.image_segmentation(ip_converted)
            processed_image = preprocessing.removebg(segmented_image)
            try :
                detect = pycolor.detect_color(processed_image,sys.argv[2])
            except:
                detect = pycolor.detect_color(processed_image,"colors.csv")
            # print (detect)
            writer.writerow([image,str(detect)])

            





    
