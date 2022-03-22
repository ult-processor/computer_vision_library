import cv2
import os 
import glob 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', required=True, help='path to the folder with images to resize. Only *.JPG files are readed'
)
args = vars(parser.parse_args())
img_dir = args['input']  
data_path = os.path.join(img_dir,'*.JPG') 
files = glob.glob(data_path) 

#== create a destination folder =====
# Directory
directory = "resized_images"
  
# Parent Directory path
parent_dir = "../"
  
# mode
mode = 0o777
  
# Path
dst_dir = os.path.join(parent_dir, directory)
# Create the directory
#with mode 0o777
os.mkdir(dst_dir, mode)
#os.mkdir(dst_dir)
print("Directory '% s' created" % directory)
#====================================
#import subprocess

#The pixel width outputS
out_width = 858  
print("Rezize image *.JPG to width size = ", out_width)
# Calculated with respect to in_width, in_height and out_width
out_height = 0 
MIN_IMG_SIZE = 10

dst_ext = "_resized.JPG"
img_count = 0
for f1 in files: 
    img = cv2.imread(f1) 
    print('Original Dimensions : ',img.shape)
    frame_width = img.shape[1]
    frame_height = img.shape[0]
    if frame_width > MIN_IMG_SIZE and frame_height > MIN_IMG_SIZE:
        aspect_ratio = float(frame_height) / float(frame_width) 
        out_height = float(out_width) * aspect_ratio
        dim = (out_width, int(out_height))
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        print('Resized Dimensions : ',resized.shape)
        cv2.imshow("Resized image", resized)
        dst_c_str = str(img_count)
        file_n = str(dst_dir + "/" + dst_c_str + dst_ext)
        print("Save image :", file_n)
        cv2.imwrite(file_n, resized)
        #subprocess.call(['chmod', mode, file_n])
        #cv2.imwrite(f1 + img_ext, resized)
        img_count = img_count + 1
    else:
        print("Error to small image frame_width = ", frame_width, "frame_height = ", frame_height)
    cv2.imshow("img", img)
    cv2.waitKey(1)
cv2.destroyAllWindows()