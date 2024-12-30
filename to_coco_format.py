import os
import json
import cv2
import numpy as np

def convert_to_coco(data_dir, output_path):
    categories = [
        {"id": 1, "name": "car"},
        {"id": 2, "name": "house"},
        {"id": 3, "name": "road"},
        {"id": 4, "name": "swimming pool"},
        {"id": 5, "name": "tree"},
        {"id": 6, "name": "yard"},
    ]
    
    images = []
    annotations = []
    annotation_id = 1

    for idx, img_name in enumerate(os.listdir(os.path.join(data_dir, "images"))):
        img_path = os.path.join(data_dir, "images", img_name)
        label_path = os.path.join(data_dir, "labels", img_name.replace(".jpg", ".txt"))

        img = cv2.imread(img_path)
        height, width, _ = img.shape

        images.append({"id": idx + 1, "file_name": img_name, "height": height, "width": width})

        with open(label_path, 'r') as file:
            for line in file.readlines():
                values = list(map(float, line.strip().split()))
                class_id = int(values[0])
                coords = np.array(values[1:]).reshape(-1, 2)

                # Denormalize
                coords[:, 0] *= width
                coords[:, 1] *= height
                coords = coords.tolist()

                # Compute bbox
                x_min, y_min = np.min(coords, axis=0)
                x_max, y_max = np.max(coords, axis=0)
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                area = (x_max - x_min) * (y_max - y_min)

                annotations.append({
                    "id": annotation_id,
                    "image_id": idx + 1,
                    "category_id": class_id + 1,  # COCO uses 1-based indexing
                    "segmentation": [np.array(coords).flatten().tolist()],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                })
                annotation_id += 1

    coco_format = {"images": images, "annotations": annotations, "categories": categories}
    
    with open(output_path, 'w') as f:
        json.dump(coco_format, f, indent=4)

# Convert dataset
convert_to_coco("./training_dataset/train", "./training_dataset/train/train.json")
convert_to_coco("./training_dataset/valid", "./training_dataset/valid/val.json")
convert_to_coco("./training_dataset/test", "./training_dataset/test/test.json")
