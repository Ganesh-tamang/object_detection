# object detection

## Installation

Clone repo and install requirements.txt

```bash
git clone git@github.com:Ganesh-tamang/object_detection.git
cd object_detection

conda create -n detectron python=3.10
conda activate detectron
# make sure to install cuda_toolkit 11.8, torch, torchvision before installing detectron
pip install -r requirements.txt  # install
```

# Download dataset
https://drive.google.com/drive/folders/1x3i-aHef5Oz1kdxUZaEX2mSSADmPp8Kl

unzip the dataset and place it at inside object detection folder
## convert the dataset to coco format
```bash
python to_coco_format.py
```

# run api
```bash
python api.py
```
## Request
```bash
curl --request POST 'http://localhost:5000/predict' --form 'image=@./training_dataset/train/images/000000095_jpg.rf.5765799a16d4712761cd7dd423be1e03.jpg'
```
