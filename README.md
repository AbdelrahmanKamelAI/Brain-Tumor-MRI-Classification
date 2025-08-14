🧠 Brain Tumor MRI Classification

This project detects brain tumors from MRI images using Deep Learning. It involves dataset balancing, augmentation, model training (CNN & Transfer Learning), and a Tkinter-based GUI application for real-time predictions.

📌 Project Workflow
1️⃣ Dataset Balancing

Initial dataset:

98 images → No Tumor

155 images → Tumor

Balancing step: Applied targeted data augmentation to increase the minority class to:

154 images → No Tumor

155 images → Tumor

2️⃣ Dataset Augmentation

Applied rotation, shifting, zooming, brightness adjustment, and flipping to generate more diverse data.

Final dataset size after augmentation:

1,426 images → No Tumor

1,428 images → Tumor

3️⃣ Model Training

Trained and compared two models:

🏗️ Custom CNN Model

Multiple convolution + pooling layers.

Optimizer: Adamax.

Loss: Categorical Crossentropy.

🔄 Transfer Learning (InceptionResNetV2)

Pre-trained on ImageNet.

Used as a frozen feature extractor + custom classification layers.

Performance:

Model	Training Accuracy	Validation Accuracy	Test Accuracy
CNN	96.67%	94.39%	98.25%
InceptionResNetV2	91.72%	92.28%	94.76%
4️⃣ GUI Application (Tkinter)

Upload MRI image.

Select model (CNN or Transfer Learning).

Instant prediction with confidence score.

Includes confusion matrix and accuracy chart visualization.

📂 Project Structure
Brain_Tumor_MRI_Classification/
│── data/                # Original & augmented datasets
│── models/              # Saved trained models
│── gui_app.py           # Tkinter application
│── train_cnn.py         # CNN model training script
│── train_transfer.py    # Transfer learning script
│── README.md            # Project documentation

🚀 How to Run

Clone the repository

git clone https://github.com/yourusername/brain-tumor-mri-classification.git
cd brain-tumor-mri-classification


Install dependencies

pip install -r requirements.txt


Run the GUI app

python gui_app.py

📷 Example GUI

(Add screenshot here)

🏆 Key Skills Demonstrated

Data preprocessing & augmentation

CNN model building from scratch

Transfer Learning (InceptionResNetV2)

Model evaluation & comparison

GUI application development with Tkinter
