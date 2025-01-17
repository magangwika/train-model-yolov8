from ultralytics import YOLO
import cv2
import math, torch

def create_video_writer(video_cap, output_filename):

    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))

    return writer

# classNames = ["APD", 'no-APD', 'person', 'vest']
classNames = ["Hardhat", "NO-Hardhat", "NO-Safety Vest", "Person", "Safety Vest"]

model = YOLO('runs/detect/yolov8s/weights/best.pt')  # Replace with your YOLOv8 model file
# model = YOLO('resources/qims-large.pt')  # Replace with your YOLOv8 model file
# model = YOLO('best.pt')  # Replace with your YOLOv8 model file

model.to('cuda')  # Explicitly move model to GPU

# Set up RTSP stream
rtsp_url = 'rtsp://admin:Bima09876@192.168.20.30:554/cam/realmonitor?channel=1&subtype=0'  # Replace with your RTSP URL
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('resources/complete.mp4')
# cap = cv2.VideoCapture('resources/sample.mp4')
# cap = cv2.VideoCapture('resources/plain.mp4')
# cap = cv2.VideoCapture(rtsp_url)
cap.set(cv2.CAP_PROP_FPS, 30)
torch.cuda.set_device(0) 


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if conf > 0.5:
                if currentClass =='NO-Helmet' or currentClass =='NO-Safety Vest' or currentClass == "NO-Hardhat":
                    myColor = (0, 0,255)
                elif currentClass =='Safety Vest' or currentClass == "Hardhat":
                    myColor =(0,255,0)
                else:
                    myColor = (255, 0, 0)


                image = cv2.putText(img, f'{classNames[cls]}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    cv2.imshow("Image", img)
#     # writer.write(img)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
# # writer.release()
cv2.destroyAllWindows()