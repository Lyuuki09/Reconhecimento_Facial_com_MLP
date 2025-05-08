# Reconhecimento Facial com MLP (PyTorch)

Este projeto implementa um sistema de reconhecimento facial utilizando redes neurais densas (MLP) em PyTorch, **sem uso de embeddings pré-treinados** e **sem CNN**. O objetivo é identificar pessoas a partir de imagens de rosto, usando apenas arquiteturas totalmente conectadas.

## Estrutura do Projeto

```
rooney-face-mlp/
│
├── data/
│   ├── raw/           # Coloque aqui as imagens originais dos rostos
│   └── processed/     # Imagens recortadas/alinhadas (geradas pelo pipeline)
│
├── models/
│   └── mlp.py         # Definição da arquitetura MLP
│
├── weights/
│   └── best_model.pth # Pesos salvos após o treinamento
│
├── utils/
│   ├── face_crop.py   # Detecção, recorte e alinhamento facial
│   └── augment.py     # Data augmentation (opcional)
│
├── train.py           # Script de treinamento
├── predict.py         # Script de inferência (identificação de rostos)
├── requirements.txt   # Dependências do projeto
├── README.md          # Instruções de uso e explicação do pipeline
└── config.yaml        # Configurações do experimento (parâmetros, caminhos, etc)
```

## Como usar

1. Coloque as imagens dos rostos em `data/raw/`, organizadas por pasta (uma pasta por pessoa).
2. Execute o pré-processamento para recortar/alinha os rostos.
3. Treine o modelo com `train.py`.
4. Use `predict.py` para identificar rostos em novas imagens.

Veja detalhes de cada etapa e exemplos de comandos nas próximas seções (em construção).

## Regras e Restrições
- Apenas MLP (redes densas), sem CNN ou embeddings pré-treinados.
- Entrada do modelo: vetor de pixels da imagem.
- Permitido usar bibliotecas de detecção facial (OpenCV, MTCNN, mediapipe).


---

## Observação sobre a pasta `venv`
A pasta `venv` contém o ambiente virtual Python utilizado para instalar as dependências do projeto localmente. **Não inclua a pasta `venv` no controle de versão (Git)**, pois ela é específica de cada máquina e pode ser recriada a qualquer momento usando o comando:

```
python -m venv venv
```

Para instalar as dependências após criar o ambiente virtual, utilize:

```
pip install -r requirements.txt
``` 