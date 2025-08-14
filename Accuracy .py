import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the models
CNN_Model = tf.keras.models.load_model(
    r"C:\Users\kabde\Desktop\ITI project\Brain Tumor MRI Classification\cnn\98\best_cnn_model.keras"
)
Transfer_Model = tf.keras.models.load_model(
    r"C:\Users\kabde\Desktop\ITI project\Brain Tumor MRI Classification\transfer\94\best_transfer_model.keras"
)

# Prepare ImageDataGenerator for the test set
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    r"C:\Users\kabde\Desktop\ITI project\Brain Tumor MRI Classification\BrainTumor\BrainTumor\brain_tumor_dataset", 
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',  
    shuffle=False
)

# Evaluate both models
loss1, acc1 = CNN_Model.evaluate(test_generator, verbose=0)
loss2, acc2 = Transfer_Model.evaluate(test_generator, verbose=0)

# Plot results as bar chart
models = ["CNN Model", "Transfer Model"]
accuracies = [acc1, acc2]

bars = plt.bar(models, accuracies, color=['skyblue', 'salmon'])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Model Comparison")

# Add percentage labels above each bar
for bar, acc in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,  # Center of the bar
        bar.get_height(),                   # Height of the bar
        f"{acc:.2%}",                       # Convert to percentage
        ha='center', va='bottom', fontsize=10, fontweight='bold'
    )

plt.show()

print(f"Model 1 Accuracy: {acc1:.2%}")
print(f"Model 2 Accuracy: {acc2:.2%}")
