import os
import cv2
from mtcnn import MTCNN
from tqdm import tqdm
import mediapipe as mp
import numpy as np


def crop_and_align_faces(input_dir, output_dir, img_size=64):
    os.makedirs(output_dir, exist_ok=True)
    detector = MTCNN()
    for person in os.listdir(input_dir):
        person_path = os.path.join(input_dir, person)
        if not os.path.isdir(person_path):
            continue
        out_person_path = os.path.join(output_dir, person)
        os.makedirs(out_person_path, exist_ok=True)
        for img_name in tqdm(os.listdir(person_path), desc=f"{person}"):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            result = detector.detect_faces(img)
            if result:
                x, y, w, h = result[0]['box']
                face = img[y:y+h, x:x+w]
                face = cv2.resize(face, (img_size, img_size))
                cv2.imwrite(os.path.join(out_person_path, img_name), face)

def preprocess_face(face_img, size=64):
    # Redimensionar
    face = cv2.resize(face_img, (size, size))
    
    # Converter para RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    
    # Normalizar
    face = face.astype(np.float32) / 255.0
    
    # Converter para vetor
    face = face.reshape(-1)
    
    return face

if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    crop_and_align_faces(
        config['data']['raw_dir'],
        config['data']['processed_dir'],
        config['data']['img_size']
    ) 