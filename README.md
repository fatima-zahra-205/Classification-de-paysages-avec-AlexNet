Ce projet réalise la classification d'images de paysages en 6 classes à l'aide du réseau de neurones profond **AlexNet**. Dataset utilisé : **Intel Image Classification Dataset** (25 000 images).

**🎯 Classes**

| **ID** | **Classe** |
| --- | --- |
| 0 | 🏢 Buildings |
| 1 | 🌲 Forest |
| 2 | 🏔️ Glacier |
| 3 | ⛰️ Mountain |
| 4 | 🌊 Sea |
| 5 | 🛣️ Street |


<img width="1536" height="1024" alt="processus" src="https://github.com/user-attachments/assets/4ccca4b9-6f00-405e-9762-dcf36e4a9c30" />


**🏗️ Architecture AlexNet**

Input (224x224x3)->

Conv1 (96 filters, 11x11, stride=4) + ReLU + MaxPool->

Conv2 (256 filters, 5x5) + ReLU + MaxPool->

Conv3 (384 filters, 3x3) + ReLU->

Conv4 (384 filters, 3x3) + ReLU->

Conv5 (256 filters, 3x3) + ReLU + MaxPool->

FC1 (4096) + Dropout + ReLU->

FC2 (4096) + Dropout + ReLU->

FC3 (6 classes)

**Total des paramètres** : ~58 millions

**📊 Résultats obtenus**

| **Métrique** | **Valeur** |
| --- | --- |
| Précision test | **84.21%** |
| Loss finale | 0.21 |

**Matrice de confusion**

<img width="598" height="284" alt="image" src="https://github.com/user-attachments/assets/d112b445-922d-47b6-8076-e3afebb6b9b9" />

**Courbes d'apprentissage**

<img width="555" height="287" alt="image" src="https://github.com/user-attachments/assets/6cff1e7e-0987-466a-bfe1-ed7e5227b413" />

Contexte du modèle : AlexNet · 6 classes (buildings, forest, glacier, mountain, sea, street) · 14 034 images train · 3 000 test · Adam · 5 epochs · EarlyStopping(patience=3)
**🛠️ Technologies utilisées**

- Python 3.13
- PyTorch 2.11
- torchvision
- Matplotlib / Seaborn
- Scikit-learn

**🚀 Installation et exécution**

**1. Cloner et installer**

pip install torch torchvision matplotlib seaborn scikit-learn pillow

**2. Télécharger le dataset**

**Dataset Intel Image Classification**

https://www.kaggle.com/datasets/puneet6060/intel-image-classification

**3. Structure des dossiers**

projet S4/ ├── archive.zip ├── intel_images/ │ └── archive1/ │ ├── seg_train/ │ │ ├── buildings/ │ │ ├── forest/ │ │ ├── glacier/ │ │ ├── mountain/ │ │ ├── sea/ │ │ └── street/ │ └── seg_test/ │ └── ... ├── alexnet_intel.pth └── training_metrics.json

**4. Lancer l'entraînement**

Exécutez le notebook Jupyter cellule par cellule.

📈 Performance par classe

Classe Précision Rappel F1-score

Buildings 0.85 0.83 0.84

Forest 0.92 0.90 0.91

Glacier 0.81 0.83 0.82

Mountain 0.79 0.81 0.80

Sea 0.86 0.88 0.87

Street 0.83 0.81 0.82

**Innovations clés d'AlexNet**

ReLU → Accélération de l'entraînement

Dropout → Réduction du sur-apprentissage

Data Augmentation → Meilleure généralisation

Utilisation du GPU → Entraînement plus rapide

**Limites**

Temps d'entraînement long sur CPU (~40 min)

58M de paramètres → modèle lourd

Sensible aux variations d'éclairage

Nécessite un resize à 224×224

**Améliorations possibles**

Fine-tuning avec un modèle pré-entraîné

Utiliser ResNet ou EfficientNet

Augmentation des données (rotation, zoom)

Early stopping pour éviter le sur-apprentissage

👤

**Auteur**

Fatima-Zahra Mahracha Projet de fin de module - Deep Learning
