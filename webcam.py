import cv2
import torch
import numpy as np
from mtcnn import MTCNN
from models.mlp import MLP
from utils.face_crop import preprocess_face

def load_model(model_path='weights/best_model.pth'):
    # Carregar o modelo
    model = MLP()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def main():
    # Inicializar detector facial
    detector = MTCNN()
    
    # Carregar modelo
    model = load_model()
    
    # Inicializar webcam
    cap = cv2.VideoCapture(0)
    
    # Classes (pessoas)
    classes = ['leandro', 'miguel', 'ygor']
    
    while True:
        # Capturar frame
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detectar faces
        faces = detector.detect_faces(frame)
        
        # Para cada face detectada
        for face in faces:
            x, y, w, h = face['box']
            
            # Extrair face
            face_img = frame[y:y+h, x:x+w]
            
            try:
                # Pré-processar face
                processed_face = preprocess_face(face_img)
                
                # Converter para tensor
                face_tensor = torch.FloatTensor(processed_face).unsqueeze(0)
                
                # Fazer predição
                with torch.no_grad():
                    outputs = model(face_tensor)
                    _, predicted = torch.max(outputs, 1)
                    confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted].item()
                
                # Obter nome da pessoa ou marcar como desconhecido
                if confidence > 0.7:
                    person = classes[predicted.item()]
                else:
                    person = 'desconhecido'
                
                # Desenhar retângulo e texto
                color = (0, 255, 0) if confidence > 0.6 else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{person} ({confidence:.2f})", 
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
            except Exception as e:
                print(f"Erro ao processar face: {e}")
                continue
        
        # Mostrar frame
        cv2.imshow('Reconhecimento Facial', frame)
        
        # Sair com 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 