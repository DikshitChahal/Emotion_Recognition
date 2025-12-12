import torch
import numpy as np
from PIL import Image
from resnet import resnet18
import os
import gdown


EMOTIONS = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]

MODEL_PATH = "https://drive.google.com/file/d/1iInqUHPgLsGaL8Brs-vv1o731b0tT9Cx/view?usp=drive_link"
DRIVE_FILE_ID = "1iInqUHPgLsGaL8Brs-vv1o731b0tT9Cx"

def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

def load_model():
    download_model()
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 7)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


def preprocess_image(image):
    image = image.convert("L").resize((48, 48))
    img = np.array(image) / 255.0
    img = (img - 0.5) / 0.5
    tensor = torch.tensor(img, dtype=torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    tensor = tensor.repeat(1, 3, 1, 1)
    return tensor

def predict_emotion(model, image):
    x = preprocess_image(image)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)
    return EMOTIONS[pred.item()], conf.item()
