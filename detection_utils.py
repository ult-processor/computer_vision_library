import cv2
from config import RESIZE_TO
def draw_bboxes(image, results, classes_to_labels):
    for image_idx in range(len(results)):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # get the original height and width of the image to resize the ...
        # ... bounding boxes to the original size size of the image
        orig_h, orig_w = image.shape[0], image.shape[1]
        # get the bounding boxes, classes, and confidence scores
        bboxes, classes, confidences = results[image_idx]
        for idx in range(len(bboxes)):
            # get the bounding box coordinates in xyxy format
            x1, y1, x2, y2 = bboxes[idx]
            # resize the bounding boxes from the normalized to RESIZE_TO pixels
            x1, y1 = int(x1*RESIZE_TO), int(y1*RESIZE_TO)
            x2, y2 = int(x2*RESIZE_TO), int(y2*RESIZE_TO)
            # resizing again to match the original dimensions of the image
            x1, y1 = int((x1/RESIZE_TO)*orig_w), int((y1/RESIZE_TO)*orig_h)
            x2, y2 = int((x2/RESIZE_TO)*orig_w), int((y2/RESIZE_TO)*orig_h)
            # draw the bounding boxes around the objects
            cv2.rectangle(
                image, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA
            )
            # put the class label text above the bounding box of each object
            cv2.putText(
                image, classes_to_labels[classes[idx]-1], (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            
    return image