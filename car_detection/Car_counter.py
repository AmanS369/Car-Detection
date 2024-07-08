from ultralytics import YOLO
import cv2
import math

id_centroid_list={}
classnames =["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag" "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat"  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush","vehicles"
            ]


def centroid_tracker(x1, y1, x2, y2, track_id):
    centroid_x = (x1 + x2) / 2
    centroid_y = (y1 + y2) / 2
    if(track_id in id_centroid_list ):
       old_x = id_centroid_list[track_id][0]
       old_y = id_centroid_list[track_id][1]
       id_centroid_list[track_id] = [centroid_x , centroid_y]
       if(abs(old_x-centroid_x)<10 and abs(old_y-centroid_y)<10 ):
          return 1
    else:
        id_centroid_list[track_id] = [centroid_x , centroid_y]
    return 0


# Open a video capture object
cap = cv2.VideoCapture("../videos/test3.mp4")
# cap.set(3, 640)
# cap.set(4,720)

model = YOLO('../Yolo_Model/yolov8l.pt')

while True:
    carcounter=0
    success, img = cap.read()
    if not success:
        # Break out of the loop if there are no more frames
        break
    results = model.track(img,stream=True,persist = True)
    for r in results:
        boxes = r.boxes
        track_id=0
        for box in boxes:
            cls = int(box.cls[0])
            currentClass = classnames[cls]
            if(currentClass == 'car'):
             print(box)
             x1,y1,x2,y2 = box.xyxy[0]
             status = "Stop"
             isMoving = 0
             if(box and box.id and box.id.item()):
              track_id=box.id.item()
              isMoving = centroid_tracker(x1,y1,x2,y2,track_id)
              if(isMoving):
                 status="Moving"


             x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)



             cv2.rectangle(img, (x1, y1, x2-x1, y2-y1), (255, 0, 255), 2)
             conf = math.ceil((box.conf[0] * 100)) / 100
             conf_text = str(conf)
             track_id=str(track_id)
             text = f"ID: {track_id} | {status}"

             cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            #  cv2.putText(img, status, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # cv2.putText(img, f'Car Count: {carcounter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Image", img)


    if cv2.waitKey(1) >= 0:
        break

# Release the video capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
