# object detection

## Installation

Clone repo and install requirements.txt

```bash
git clone 
conda create -n detectron python=3.10
conda activate detectron
cd 
pip install -r requirements.txt  # install
```

# Download dataset
https://drive.google.com/drive/folders/1x3i-aHef5Oz1kdxUZaEX2mSSADmPp8Kl

# run api
```bash
python api.py
```
## Request
```bash
curl --request POST 'http://localhost:5000/predict' --form 'image=@./training_dataset/train/images/000000095_jpg.rf.5765799a16d4712761cd7dd423be1e03.jpg'
```