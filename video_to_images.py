import cv2
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', required=True, help='path to the input video'
)
args = vars(parser.parse_args())

# capture the video
cap = cv2.VideoCapture(args['input'])
#cap = cv2.VideoCapture(0)
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')

img_dir = "../images_from_video/"
img_ext = ".jpg"
img_count = 0
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('extracted_images', frame)
        #out.write(orig_image)
        img_c_str = str(img_count)
        cv2.imwrite(img_dir + img_c_str + img_ext, frame)
        cv2.waitKey(1)
        img_count = img_count + 1
    else:
        break
# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()