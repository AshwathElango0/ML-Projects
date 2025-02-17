import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from collections import defaultdict

# 1. Configuration
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_path = "fasterrcnn_drone.pth"  # Path to the saved model
num_classes = 2  # Number of classes (background + drone)
threshold = 0.5  # Confidence threshold for predictions
iou_threshold = 0.3  # IOU threshold for tracking

# 2. Load Model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 3. Helper Function: Compute IOU (Intersection over Union)
def compute_iou(box1, box2):
    """Calculate the Intersection over Union (IOU) between two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    # Compute the intersection
    xA = max(x1, x1_p)
    yA = max(y1, y1_p)
    xB = min(x2, x2_p)
    yB = min(y2, y2_p)
    intersection = max(0, xB - xA) * max(0, yB - yA)

    # Compute the union
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0

# 4. Tracking Class
class ObjectTracker:
    def __init__(self, iou_threshold=0.3):
        self.next_id = 0
        self.tracks = defaultdict(dict)  # Track ID to bounding box & confidence
        self.iou_threshold = iou_threshold

    def update(self, detections):
        """Update the tracker with new detections."""
        new_tracks = defaultdict(dict)

        for detection in detections:
            matched = False
            for track_id, track in self.tracks.items():
                iou = compute_iou(track["box"], detection["box"])
                if iou >= self.iou_threshold:
                    new_tracks[track_id] = detection
                    matched = True
                    break

            if not matched:  # Assign a new ID
                new_tracks[self.next_id] = detection
                self.next_id += 1

        self.tracks = new_tracks
        return self.tracks

# 5. Initialize Tracker
tracker = ObjectTracker(iou_threshold=iou_threshold)

# 6. Frame Annotation with Tracking
def annotate_frame(frame, model, tracker, threshold=0.5):
    # Convert OpenCV frame (BGR) to PIL Image (RGB)
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = F.to_tensor(pil_image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        predictions = model(img_tensor)[0]

    detections = []
    for box, label, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"]):
        if score >= threshold:
            detections.append({"box": box.tolist(), "label": label.item(), "score": score.item()})

    # Update tracker
    tracks = tracker.update(detections)

    # Draw bounding boxes and labels on the image
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default()

    for track_id, track in tracks.items():
        xmin, ymin, xmax, ymax = track["box"]
        label_text = f"ID {track_id}: Class {track['label']} ({track['score']:.2f})"
        draw.rectangle([xmin, ymin, xmax, ymax], outline="blue", width=3)
        draw.text((xmin, ymin), label_text, fill="blue", font=font)

    # Convert PIL Image back to OpenCV format (BGR)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# 7. OpenCV Live Video Capture with Tracking
def live_video_annotation():
    cap = cv2.VideoCapture(0)  # Open the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to exit the live video annotation with tracking.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Annotate the current frame
        annotated_frame = annotate_frame(frame, model, tracker, threshold=threshold)

        # Display the annotated frame
        cv2.imshow("Live Video Annotation with Tracking", annotated_frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# 8. Run the Live Annotation Application
if __name__ == "__main__":
    live_video_annotation()
