import torch
import numpy as np
from PIL import Image
from resnet import resnet18

EMOTIONS = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]

def load_model():
    model = resnet18(num_classes=7)
    state_dict = torch.load(
        "emotion_resnet18_weights.pth",
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(state_dict)
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
