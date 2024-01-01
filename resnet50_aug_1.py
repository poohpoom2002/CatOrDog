# %%
# Import necessary libraries
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

# Define your parameters
batch_size = 8
epoch = 150
node_size = 512
learning_rate = 0.0001
isFreeze = True
image_size = (300, 300)
NUM_CLASSES = 2

# Load the ResNet-50 model without the top classification layer


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(300, 300, 3))

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(node_size, activation='relu')(x)
x = Dense(node_size/2, activation='relu')(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# Create the new model
model = Model(inputs=base_model.input, outputs=predictions)

# Optionally freeze the layers of the base model
if isFreeze:
    for layer in base_model.layers:
        layer.trainable = False

# Define data augmentation parameters
rotation_range = 30
zoom_range = 0.2
width_shift_range = 0.2
height_shift_range = 0.2
shear_range = 0.2
horizontal_flip = True
seed_value = 42

# Create DataGenerator Object
datagen = ImageDataGenerator(
    rotation_range=rotation_range,
    zoom_range=zoom_range,
    width_shift_range=width_shift_range,
    height_shift_range=height_shift_range,
    shear_range=shear_range,
    horizontal_flip=horizontal_flip,
    fill_mode="nearest"
)



# Create Data Generators
train_generator = datagen.flow_from_directory(
    "./data/train_aug/",
    target_size=image_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    seed=seed_value,
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    "./data/validate/",
    target_size=image_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    seed=seed_value,
    shuffle=True
)

test_generator = datagen.flow_from_directory(
    "./data/test/",
    target_size=image_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    seed=seed_value,
    shuffle=True
)

# Compile the model
opts = Adadelta(learning_rate=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=opts, metrics=['accuracy'])

# Calculate step sizes for training and validation generators
step_size_train = len(train_generator)
step_size_val = len(val_generator)

# %%
# Create an Early Stopping callback
callback = EarlyStopping(monitor='loss', patience=10)

# Train the model
history = model.fit_generator(generator=train_generator,
                            steps_per_epoch=step_size_train,
                            validation_data=val_generator,
                            validation_steps=step_size_val,
                            epochs=epoch,
                            callbacks=[callback],
                            verbose=1)

# Extract the training history
train_accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
train_loss = history.history["loss"]
val_loss = history.history["val_loss"]

# Create a list of epochs for plotting
N = list(range(1, len(train_accuracy) + 1))


# View Accuracy (Training, Validation)
plt.plot(N, train_accuracy, label="Train Accuracy")
plt.plot(N, val_accuracy, label="Validation Accuracy")
plt.title(f"Accuracy vs. Epoch {epoch}")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(f'result/aug_acc_bs{batch_size}_e{epoch}_color_lr{learning_rate}_nn{node_size}_c{NUM_CLASSES}_fr{isFreeze}.png')
plt.close()

# View Loss (Training, Validation)
plt.plot(N, train_loss, label="Train Loss")
plt.plot(N, val_loss, label="Validation Loss")
plt.title(f"Loss vs. Epoch {epoch}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f'result/aug_lost_bs{batch_size}_e{epoch}_color_lr{learning_rate}_nn{node_size}_c{NUM_CLASSES}_fr{isFreeze}.png')
plt.close()

# Print additional information
print("Batch size:", batch_size)
print("Epochs:", epoch)
print("Number of nodes:", node_size)

# Evaluate the model on the test data
y_true = test_generator.classes
preds = model.predict_generator(test_generator)
y_pred = np.argmax(preds, axis=1)
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("Classification Report:")
print(classification_report(y_true, y_pred))

# Display the plots
plt.tight_layout()
plt.show()

# # Save the model
model.save(f'model/aug_inception_bs{batch_size}_e{epoch}_lr{learning_rate}_nn{node_size}_c{NUM_CLASSES}_fr{isFreeze}.h5')


