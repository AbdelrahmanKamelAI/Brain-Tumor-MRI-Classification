import tkinter as tk
from tkinter import filedialog, messagebox
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageTk

# Load the models
cnn_model = load_model(r"C:\Users\kabde\Desktop\ITI project\Brain Tumor MRI Classification\cnn\98\best_cnn_model.keras")
transfer_model = load_model(r"C:\Users\kabde\Desktop\ITI project\Brain Tumor MRI Classification\transfer\94\best_transfer_model.keras")

# Setup the main window
root = tk.Tk()
root.title("Brain Tumor Detection")
root.geometry("900x600")
root.config(bg="#f0f4f7")

selected_model = tk.StringVar(value="CNN")
image_path = None

# Title label
title_label = tk.Label(root, text="🧠 Brain Tumor Detection", font=("Arial", 20, "bold"), bg="#f0f4f7", fg="#2c3e50")
title_label.pack(pady=10)

# Main layout
main_frame = tk.Frame(root, bg="#f0f4f7")
main_frame.pack(fill="both", expand=True, padx=20, pady=10)

left_frame = tk.Frame(main_frame, bg="#f0f4f7")
left_frame.pack(side="left", fill="y", padx=10)

right_frame = tk.Frame(main_frame, bg="#f0f4f7")
right_frame.pack(side="right", fill="both", expand=True, padx=10)

# ----------------- Functions -----------------
def upload_image():
    """Open file dialog to select an image and display it."""
    global image_path
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        image_path = file_path
        img = Image.open(file_path)
        img.thumbnail((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        model_image_label.config(image=img_tk)
        model_image_label.image = img_tk
        result_label.config(text="")

def predict_image():
    """Predict tumor presence using the selected model."""
    global image_path
    if not image_path:
        messagebox.showerror("Error", "Please upload an image first")
        return
    
    # Open image and ensure it's RGB with 3 channels
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalization
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    
    model = cnn_model if selected_model.get() == "CNN" else transfer_model
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100
    
    labels = ['no', 'yes']
    result = labels[class_idx]
    
    result_label.config(text=f"Predicted Tumor: {result}\nConfidence: {confidence:.2f}%", fg="#27ae60")

# Image display functions
def show_fixed1():
    img = Image.open(fixed_image1_path)
    img.thumbnail((400, 400))
    img_tk = ImageTk.PhotoImage(img)
    fixed_img1_label.config(image=img_tk)
    fixed_img1_label.image = img_tk

def show_fixed2():
    img = Image.open(fixed_image2_path)
    img.thumbnail((400, 400))
    img_tk = ImageTk.PhotoImage(img)
    fixed_img2_label.config(image=img_tk)
    fixed_img2_label.image = img_tk

def show_fixed3():
    img = Image.open(fixed_image3_path)
    img.thumbnail((400, 400))
    img_tk = ImageTk.PhotoImage(img)
    fixed_img3_label.config(image=img_tk)
    fixed_img3_label.image = img_tk

# Auto update images when model changes
def update_images():
    if selected_model.get() == "CNN":
        show_img1_var.set(True)
        show_fixed1()
        show_img3_var.set(True)
        show_fixed3()
        show_img2_var.set(False)
        fixed_img2_label.config(image="")
    else:
        show_img2_var.set(True)
        show_fixed2()
        show_img3_var.set(True)
        show_fixed3()
        show_img1_var.set(False)
        fixed_img1_label.config(image="")

# ----------------- UI Layout -----------------
# Model selection
tk.Label(left_frame, text="Select Model:", font=("Arial", 14), bg="#f0f4f7", fg="#34495e").pack(anchor="w", pady=5)
tk.Radiobutton(left_frame, text="CNN Model", variable=selected_model, value="CNN", font=("Arial", 12),
               bg="#f0f4f7", fg="#2c3e50", selectcolor="#d6eaf8", command=update_images).pack(anchor="w")
tk.Radiobutton(left_frame, text="Transfer Learning Model", variable=selected_model, value="Transfer", font=("Arial", 12),
               bg="#f0f4f7", fg="#2c3e50", selectcolor="#d6eaf8", command=update_images).pack(anchor="w")

# Upload & Predict buttons
tk.Button(left_frame, text="📂 Upload Image", command=upload_image, font=("Arial", 12, "bold"),
          bg="#3498db", fg="white", activebackground="#2980b9", activeforeground="white",
          relief="raised", bd=3).pack(pady=15, fill="x")

tk.Button(left_frame, text="🔍 Predict", command=predict_image, font=("Arial", 12, "bold"),
          bg="#2ecc71", fg="white", activebackground="#27ae60", activeforeground="white",
          relief="raised", bd=3).pack(pady=5, fill="x")

# Checkbuttons for images
show_img1_var = tk.BooleanVar()
show_img2_var = tk.BooleanVar()
show_img3_var = tk.BooleanVar()

tk.Checkbutton(left_frame, text="Confusion matrix for CNN", variable=show_img1_var, command=lambda: show_fixed1() if show_img1_var.get() else fixed_img1_label.config(image=""),
               font=("Arial", 12), bg="#f0f4f7", fg="#2c3e50", selectcolor="#d6eaf8").pack(anchor="w")
tk.Checkbutton(left_frame, text="Confusion matrix for Transfer Learning", variable=show_img2_var, command=lambda: show_fixed2() if show_img2_var.get() else fixed_img2_label.config(image=""),
               font=("Arial", 12), bg="#f0f4f7", fg="#2c3e50", selectcolor="#d6eaf8").pack(anchor="w")
tk.Checkbutton(left_frame, text="Accuracy", variable=show_img3_var, command=lambda: show_fixed3() if show_img3_var.get() else fixed_img3_label.config(image=""),
               font=("Arial", 12), bg="#f0f4f7", fg="#2c3e50", selectcolor="#d6eaf8").pack(anchor="w")

# Right frame: uploaded image preview + prediction
model_image_label = tk.Label(right_frame, bg="#f0f4f7")
model_image_label.pack(pady=10)

result_label = tk.Label(right_frame, text="", font=("Arial", 14, "bold"), bg="#f0f4f7", fg="#2c3e50")
result_label.pack(pady=10)

# Fixed images section
tk.Label(right_frame, text="Confusion matrix and Accuracy", font=("Arial", 14, "bold"), bg="#f0f4f7", fg="#34495e").pack(pady=10)
fixed_images_frame = tk.Frame(right_frame, bg="#f0f4f7")
fixed_images_frame.pack(pady=5)

fixed_image1_path = "cnn\\98\\CNN.png"
fixed_image2_path = "transfer\\94\\Transfer Learning.png"
fixed_image3_path = "Accuracy/Accuracy.png"

fixed_img1_label = tk.Label(fixed_images_frame, bg="#f0f4f7")
fixed_img1_label.grid(row=0, column=0, padx=10)

fixed_img2_label = tk.Label(fixed_images_frame, bg="#f0f4f7")
fixed_img2_label.grid(row=0, column=1, padx=10)

fixed_img3_label = tk.Label(fixed_images_frame, bg="#f0f4f7")
fixed_img3_label.grid(row=0, column=2, padx=10)

# Default CNN images on start
update_images()

root.mainloop()
