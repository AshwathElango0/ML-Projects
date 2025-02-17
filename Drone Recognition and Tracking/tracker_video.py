import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
import numpy as np

# Configuration
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_path = "fasterrcnn_drone.pth" # Replace with path to trained model
num_classes = 2  # Background + Drone
threshold = 0.5  # Confidence threshold
input_video_path = r""  # Replace with path to input video
output_video_path = "annotated_output_video.avi"
max_disappeared_frames = 10
iou_threshold = 0.3

# Load model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

# Video capture and output
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Error: Cannot open video file {input_video_path}")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Object tracking structures
tracked_objects = {}
next_object_id = 0

def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    inter_x1 = max(x1, x1g)
    inter_y1 = max(y1, y1g)
    inter_x2 = min(x2, x2g)
    inter_y2 = min(y2, y2g)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        intersection = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    else:
        intersection = 0

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2g - x1g) * (y2g - y1g)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def update_tracked_objects(detected_boxes):
    """Update tracked objects using IoU."""
    global tracked_objects, next_object_id

    matched_ids = set()
    updated_objects = {}

    for object_id, tracked_box in tracked_objects.items():
        best_iou = 0
        best_box = None

        for i, detected_box in enumerate(detected_boxes):
            iou = compute_iou(tracked_box["box"], detected_box)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_box = (i, detected_box)

        if best_box:
            idx, new_box = best_box
            updated_objects[object_id] = {"box": new_box, "disappeared": 0}
            matched_ids.add(idx)

    # Register new objects
    for i, box in enumerate(detected_boxes):
        if i not in matched_ids:
            updated_objects[next_object_id] = {"box": box, "disappeared": 0}
            next_object_id += 1

    # Increment disappearance count for unmatched objects
    for object_id in tracked_objects:
        if object_id not in updated_objects:
            tracked_box = tracked_objects[object_id]
            tracked_box["disappeared"] += 1
            if tracked_box["disappeared"] <= max_disappeared_frames:
                updated_objects[object_id] = tracked_box

    tracked_objects = updated_objects

# Process video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        print("Finished processing video.")
        break

    # Convert frame to model input
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img).unsqueeze(0).to(device)

    # Object detection
    with torch.no_grad():
        predictions = model(img_tensor)[0]

    # Filter predictions
    detected_boxes = [
        box.cpu().numpy()
        for box, score in zip(predictions["boxes"], predictions["scores"])
        if score >= threshold
    ]

    # Update tracking
    update_tracked_objects(detected_boxes)

    # Annotate frame
    for object_id, tracked_box in tracked_objects.items():
        xmin, ymin, xmax, ymax = map(int, tracked_box["box"])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"ID {object_id}",
            (xmin, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    # Write annotated frame to output
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Annotated video saved as {output_video_path}")
