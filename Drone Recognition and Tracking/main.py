import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 1. Configuration
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model_path = "fasterrcnn_drone.pth"  # Path to the saved model
num_classes = 2  # Number of classes (background + drone)
threshold = 0.5  # Confidence threshold for predictions

# 2. Load Model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 3. Helper Function for Annotation
def annotate_frame(frame, model, threshold=0.5):
    # Convert OpenCV frame (BGR) to PIL Image (RGB)
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img_tensor = F.to_tensor(pil_image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        predictions = model(img_tensor)[0]

    # Draw bounding boxes and labels on the image
    draw = ImageDraw.Draw(pil_image)
    font = ImageFont.load_default()

    for box, label, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"]):
        if score >= threshold:
            xmin, ymin, xmax, ymax = box.tolist()
            label_text = f"Class {label} ({score:.2f})"
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
            draw.text((xmin, ymin), label_text, fill="red", font=font)

    # Convert PIL Image back to OpenCV format (BGR)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# 4. OpenCV Live Video Capture
def live_video_annotation():
    cap = cv2.VideoCapture(0)  # Open the default webcam (use 0 for default camera)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to exit the live video annotation.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Annotate the current frame
        annotated_frame = annotate_frame(frame, model, threshold=threshold)

        # Display the annotated frame
        cv2.imshow("Live Video Annotation", annotated_frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# 5. Run the Live Annotation Application
if __name__ == "__main__":
    live_video_annotation()
