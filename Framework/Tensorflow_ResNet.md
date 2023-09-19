
```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy

# 1. Data Loading and Preprocessing
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize pixel values between 0 and 1
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Convert labels to one-hot encoded vectors
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

# 2. Model Definition (ResNet Block and Full Model)
def resnet_block(inputs, filters, kernel_size=3, stride=1, activation="relu"):
    x = Conv2D(filters, kernel_size, strides=stride, padding="same")(inputs)
    x = BatchNormalization()(x)
    if activation:
        x = ReLU()(x)
    x = Conv2D(filters, kernel_size, strides=stride, padding="same")(x)
    x = BatchNormalization()(x)
    
    # Add the input features to the block's output to form the residual connection
    x = Add()([x, inputs])
    if activation:
        x = ReLU()(x)
    return x

# Define the full ResNet model
inputs = Input(shape=(32, 32, 3))
x = Conv2D(32, 3, activation="relu")(inputs)
x = resnet_block(x, 32)
x = resnet_block(x, 32)
x = GlobalAveragePooling2D()(x)
outputs = Dense(10, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)

# 3. Model Compilation
model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=[CategoricalAccuracy()])

# 4. Model Training
model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))

# 5. Model Evaluation
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Optionally, save the model for future use
# model.save("resnet_model.h5")
```

### Explanation:

1. **Data Loading and Preprocessing**: We load the CIFAR-10 dataset, normalize the images, and convert the labels to one-hot encoded vectors.

2. **Model Definition**: We define a ResNet block function that creates a residual block. We then use this function to define the full ResNet model. This example uses a very simplified version of ResNet for brevity.

3. **Model Compilation**: We specify the optimizer, loss function, and metrics for training.

4. **Model Training**: We train the model using the training data and validate it using the test data.

5. **Model Evaluation**: We evaluate the trained model on the test dataset to get the final loss and accuracy.

This example provides a basic workflow for training a ResNet model using TensorFlow. In practice, you might include additional components like data augmentation, learning rate schedules, callbacks, and more advanced ResNet architectures.
