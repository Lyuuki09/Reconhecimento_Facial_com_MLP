import os
import yaml
import torch
import numpy as np
import cv2
import pickle
from models.mlp import FaceMLP

def load_config():
    with open('config.yaml') as f:
        return yaml.safe_load(f)

def preprocess_image(img_path, img_size):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Imagem não encontrada: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1)).reshape(-1)
    return torch.tensor(img).unsqueeze(0)

def predict(img_path):
    config = load_config()
    input_dim = config['data']['img_size'] ** 2 * 3
    # Carrega modelo
    with open('weights/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    num_classes = len(le.classes_)
    model = FaceMLP(input_dim, num_classes, config['model']['hidden_layers'], config['model']['dropout'])
    model.load_state_dict(torch.load(config['weights']['save_path'], map_location='cpu'))
    model.eval()
    # Preprocessa imagem
    x = preprocess_image(img_path, config['data']['img_size'])
    # Predição
    with torch.no_grad():
        out = model(x)
        pred = out.argmax(1).item()
        label = le.inverse_transform([pred])[0]
    print(f"Predição: {label}")
    return label

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python predict.py caminho/da/imagem.jpg")
    else:
        predict(sys.argv[1]) 