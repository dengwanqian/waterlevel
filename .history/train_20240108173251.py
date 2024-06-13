import cv2
import numpy as np

# Load the training data
train_data = np.load('train_data.npy')
train_labels = np.load('train_labels.npy')

# Create a k-Nearest Neighbors classifier
knn = cv2.ml.KNearest_create()

# Train the classifier
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

# Load the test image
test_image = cv2.imread('test_image.jpg')

# Preprocess the test image
# ... perform any necessary preprocessing steps, such as resizing or normalizing ...

# Reshape the test image to match the training data shape
test_data = test_image.reshape(1, -1).astype(np.float32)

# Use the classifier to predict the label of the test image
_, result, _, _ = knn.findNearest(test_data, k=1)

# Print the predicted label
print("Predicted label:", result[0][0])