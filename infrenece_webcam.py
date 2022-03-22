import numpy as np
import cv2
from numpy.core.fromnumeric import size
import torch
import glob as glob

import time
from config import RESIZE_TO
from config import CLASSES
from config import MODEL_DICT_FILE
from config import OUT_DIR
from detection_utils import draw_bboxes
from model import create_model
import argparse
cap = cv2.VideoCapture(0)
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')
# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# define codec and create VideoWriter object 
#out = cv2.VideoWriter(f"../outputs/webcam.mp4", 
#                      cv2.VideoWriter_fourcc(*'mp4v'), 20, 
#                      (frame_width, frame_height))

out2 = cv2.VideoWriter(f"../outputs/webcam.avi", 
                      cv2.VideoWriter_fourcc(*'XVID'), 20, 
                      (frame_width, frame_height))


frame_count = 0 # to count total frames
total_fps = 0 # to get the final frames per second

# set the computation device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# load the model and the trained weights
model = create_model(num_classes=5).to(device)
model.load_state_dict(torch.load(
    MODEL_DICT_FILE, map_location=device
))
model.eval()

# directory where all the images are present
DIR_TEST = '../test_data'
test_images = glob.glob(f"{DIR_TEST}/*")
print(f"Test instances: {len(test_images)}")


# define the detection threshold...
detection_threshold = 0.75

# read until end of video
pre_end_time = 1.0
pre_start_time = 0.0
while(cap.isOpened()):
    # get the start time
    start_time = time.time()
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        # convert frame
        #image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #image = cv2.imread(test_images[i])
        orig_image = frame.copy()
        # BGR to RGB
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(np.float)
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float).cuda()
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            outputs = model(image)
        
        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        fps = 1
        # carry further only if there are detected boxes
        if len(outputs[0]['boxes']) != 0:
            
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            draw_boxes = boxes.copy()
            # get all the predicited class names
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
            
            # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):
                text_red = list([0,0,255])
                text_green = list([50,255,10])
                text_color = list([255,0,0])
                if pred_classes[j] == CLASSES[1]:
                    text_color = text_red
                if pred_classes[j] == CLASSES[2]:
                    text_color = text_green

                cv2.rectangle(orig_image,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (0, 0, 255), 2)
                cv2.putText(orig_image, pred_classes[j], 
                            (int(box[0])+10, int(box[1]+20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (text_color), 
                            2, lineType=cv2.LINE_AA)
                cv2.putText(orig_image, str("{:.2f}".format(scores[j])), 
                            (int(box[0])+10, int(box[1]+55)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.90, (255, 20, 20), 
                            1, lineType=cv2.LINE_AA)
            # get the fps
            fps = 1 / (pre_end_time - pre_start_time)
            # add fps to total fps
            total_fps += fps
            # increment frame count
            frame_count += 1
            # write the FPS on current frame
            cv2.putText(
                orig_image, f"{fps:.3f} FPS", (5, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),
                2
            )
            cv2.imshow('Prediction', orig_image)
            #out.write(orig_image)
            out2.write(orig_image)
            cv2.waitKey(1)
            # get the end time
            pre_end_time = time.time()
            pre_start_time = start_time
    else:
        break
# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()
# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")

print('END.. ')
cv2.destroyAllWindows()