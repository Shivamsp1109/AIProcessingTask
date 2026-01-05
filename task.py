import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# Load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Resize to 28x28 and normalize
train_images = train_images.reshape(-1, 28, 28, 1).astype("float32") / 255.0
test_images  = test_images.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Build simple CNN model
model = models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
print("Training model...")
model.fit(train_images, train_labels, epochs=5, batch_size=64, verbose=1)

# Evaluate
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

# Prediction on a sample image
i = np.random.randint(0, len(test_images))
sample = test_images[i]
prediction = np.argmax(model.predict(sample.reshape(1,28,28,1)))
print("Predicted class:", prediction)

# Display image and result
img = sample.squeeze() * 255
img = img.astype("uint8")
img = cv2.resize(img, (280, 280), interpolation=cv2.INTER_NEAREST)
cv2.putText(img, f"Prediction: {prediction}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

cv2.imshow("Sample Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()