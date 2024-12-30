from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

app = Flask(__name__)

# Initialize the Detectron2 model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # threshold for the model
cfg.MODEL.WEIGHTS = "output/model_final.pth"  # trained model weights path
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6  # number of class

thing_classes = ["car", "house", "road", "swimming pool", "tree", "yard"]

# predictor
predictor = DefaultPredictor(cfg)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure an image is included in the request
        if 'image' not in request.files:
            return jsonify({"error": "No image file found in the request"}), 400

        # Read the image file
        image_file = request.files['image']
        image_np = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        # Run prediction
        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        image_height, image_width = image.shape[0], image.shape[1]
        # Extract predictions
        predictions = []
        for i in range(len(instances)):  # Iterate over detected instances
            bbox = instances.pred_boxes[i].tensor.numpy().tolist()[0]
            label_idx = int(instances.pred_classes[i].item())
            confidence = float(instances.scores[i].item())
            label = thing_classes[label_idx]
            x_min, y_min, x_max, y_max = bbox
            x_min_norm = x_min / image_width
            y_min_norm = y_min / image_height
            x_max_norm = x_max / image_width
            y_max_norm = y_max / image_height
    
            predictions.append({
                "label": label,
                "confidence": confidence,
                "bbox": [x_min_norm, y_min_norm, x_max_norm, y_max_norm],
                
            })

        return jsonify({"prediction": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
