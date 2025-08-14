🧠 Brain Tumor MRI Classification

A deep learning project to classify brain MRI scans as Tumor or No Tumor, with dataset balancing, augmentation, model training, and a Tkinter GUI for predictions.

📌 Steps

Balance Data

From 98/155 → Balanced to 154/155.

Augment Data

Increased to ~1426/1428 images using rotation, flip, zoom, brightness changes.

Train Models

Custom CNN and Transfer Learning (InceptionResNetV2).

GUI App

Upload image → Select model → Get prediction.

📊 Accuracy
Model	Test Accuracy
CNN	98.25%
Transfer Learning	94.76%
