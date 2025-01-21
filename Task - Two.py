try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
    from tensorflow.keras.datasets import cifar10
    import matplotlib.pyplot as plt
    import numpy as np
except ModuleNotFoundError as e:
    print("Required modules are missing. Please ensure TensorFlow and other dependencies are installed.")
    raise e

# Load and preprocess CIFAR-10 dataset
try:
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
except Exception as e:
    print("Error loading the CIFAR-10 dataset.")
    raise e

# Normalize the pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Define class names
class_names = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

# Display some sample images
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_train[i])
    plt.title(class_names[y_train[i][0]])
    plt.axis('off')
plt.show()

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
try:
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
except Exception as e:
    print("Error during training.")
    raise e

# Evaluate the model
try:
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.2f}, Test Accuracy: {accuracy:.2f}")
except Exception as e:
    print("Error during evaluation.")
    raise e

# Plot training and validation accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

# Make predictions on the test set
try:
    predictions = model.predict(X_test)
except Exception as e:
    print("Error during predictions.")
    raise e

# Visualize some predictions
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[i])
    pred_label = class_names[np.argmax(predictions[i])]
    true_label = class_names[y_test[i][0]]
    plt.title(f"Pred: {pred_label}\nTrue: {true_label}")
    plt.axis('off')
plt.show()

# Save the trained model
try:
    model.save('cifar10_cnn_model.h5')
    print("Model saved successfully as 'cifar10_cnn_model.h5'.")
except Exception as e:
    print("Error saving the model.")
    raise e
