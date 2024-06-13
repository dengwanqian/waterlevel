import os
import cv2
import numpy as np


# Define the directory path where the training images are located
directory = 'training_images/'

# Initialize empty lists to store the image data and labels
image_data = []
labels = []

# Iterate over each image file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Read the image file
        image = cv2.imread(os.path.join(directory, filename))
        
        # Preprocess the image as needed
        # ... perform any necessary preprocessing steps, such as resizing or normalizing ...
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Flatten the image array and append it to the image data list
        image_data.append(gray.flatten())
        
        # Extract the label from the filename (assuming the filename format is 'label_image.jpg')
        label = filename.split('_')[0]
        labels.append(label)

'''

In this example, we assume that the training images are stored in a directory called 'training_images/'. You can modify the directory variable to match the actual path of your image directory.

The code uses a for loop to iterate over each image file in the directory. It reads each image using cv2.imread() and performs any necessary preprocessing steps (e.g., resizing, normalizing, or converting to grayscale).

The image data is then flattened using the flatten() method to convert it into a 1D array, and the flattened array is appended to the image_data list.

The label is extracted from the filename assuming that the filename format is 'label_image.jpg', where the label is the prefix before the underscore.

Finally, the image data and labels are converted to NumPy arrays using np.array(), and they are saved as .npy files using np.save() for future use.

Remember to import the necessary libraries at the beginning of your script, such as cv2, numpy, and os.
'''        

# Convert the image data and labels lists to NumPy arrays
image_data = np.array(image_data)
labels = np.array(labels)

# Save the image data and labels as .npy files for future use
np.save('train_data.npy', image_data)
np.save('train_labels.npy', labels)


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

