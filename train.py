import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from models.mlp import FaceMLP

def augment_image(img):
    """Aplica augmentation na imagem."""
    augmented = [img]
    # Rotação
    for angle in [15, -15]:
        M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle, 1)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        augmented.append(rotated)
    # Flip horizontal
    flipped = cv2.flip(img, 1)
    augmented.append(flipped)
    # Ajuste de brilho
    bright = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
    dark = cv2.convertScaleAbs(img, alpha=0.8, beta=-10)
    augmented.extend([bright, dark])
    return augmented

# Dataset customizado
class FaceDataset(Dataset):
    def __init__(self, data_dir, img_size, augment=False):
        self.imgs = []
        self.labels = []
        self.img_size = img_size
        self.augment = augment
        for person in os.listdir(data_dir):
            person_path = os.path.join(data_dir, person)
            if not os.path.isdir(person_path):
                continue
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size))
                if augment:
                    for aug_img in augment_image(img):
                        self.imgs.append(aug_img)
                        self.labels.append(person)
                else:
                    self.imgs.append(img)
                    self.labels.append(person)
        self.imgs = np.array(self.imgs, dtype=np.float32) / 255.0
        self.imgs = self.imgs.transpose(0, 3, 1, 2)  # NCHW
        self.imgs = self.imgs.reshape(len(self.imgs), -1)  # Flatten
        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(self.labels)
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        return torch.tensor(self.imgs[idx]), torch.tensor(self.labels[idx])
    def get_num_classes(self):
        return len(self.le.classes_)
    def get_label_encoder(self):
        return self.le

def train():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    torch.manual_seed(config['train']['seed'])
    # Dataset
    dataset = FaceDataset(
        config['data']['processed_dir'],
        config['data']['img_size'],
        augment=True
    )
    num_classes = dataset.get_num_classes()
    input_dim = config['data']['img_size'] ** 2 * 3
    # Split
    val_size = int(len(dataset) * config['train']['val_split'])
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=config['train']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['train']['batch_size'])
    # Modelo
    model = FaceMLP(input_dim, num_classes, config['model']['hidden_layers'], config['model']['dropout'])
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # Loss e otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['train']['lr'])
    best_acc = 0
    for epoch in range(config['train']['epochs']):
        model.train()
        running_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # Validação
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                preds = out.argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"Val Acc: {acc:.4f} | Loss: {running_loss/len(train_loader):.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), config['weights']['save_path'])
            print("[!] Modelo salvo com melhor acurácia.")
    # Salva label encoder
    import pickle
    with open('weights/label_encoder.pkl', 'wb') as f:
        pickle.dump(dataset.get_label_encoder(), f)

if __name__ == "__main__":
    train() 