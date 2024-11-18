
import cv2
import torch
import numpy as np

# Load the YOLOv5 model
model = torch.hub.load('/home/lakshmanan/parkingspace/yolov5', 'custom',
                       path='/home/lakshmanan/parkingspace/model/best.pt',
                       source='local')

# Function to process detections and draw bounding boxes
def process_detections(frame, results, confidence_threshold=0.3):
    total_spaces = 0
    filled_spaces = 0
    data = []

    for result in results.xyxy[0]:  # xyxy format: [x1, y1, x2, y2, confidence, class]
        x1, y1, x2, y2, confidence, cls = result.cpu().numpy()

        if confidence < confidence_threshold:
            continue  # Ignore detections below confidence threshold

        cls = int(cls)
        if cls == 0:  # Parking space class (free space)
            total_spaces += 1
            data.append(0)  # 0 means no car parked
            color = (0, 0, 255)  # Red for free spaces
        elif cls == 1:  # Car class (occupied space)
            total_spaces += 1
            filled_spaces += 1
            data.append(1)  # 1 means car parked
            color = (0, 255, 0)  # Green for cars (occupied spaces)
        else:
            continue  # Ignore other classes

        label = f"{model.names[cls]} {confidence:.2f}"

        # Draw rectangle and label
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    not_filled_spaces = total_spaces - filled_spaces
    output = {
        "Total spaces": total_spaces,
        "Filled": filled_spaces,
        "Not Filled": not_filled_spaces,
        "Data": data
    }

    return frame, output

# OpenCV Video Capture (0 for webcam, or replace with video file path)
cap = cv2.VideoCapture(0)  # Change to 'rtsp://your_ip_camera_url' for IP camera feed

if not cap.isOpened():
    print("Error: Unable to access the camera or video feed.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to retrieve frame from feed.")
        break

    # Preprocess the frame for YOLO model
    results = model(frame)

    # Process detections and draw bounding boxes
    frame_with_boxes, parking_data = process_detections(frame, results)

    # Print parking data to the console
    print(parking_data)

    # Display the frame with detections
    cv2.imshow("Parking Space Detection", frame_with_boxes)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

