# Importing necessary libraries
import tensorflow as tf
import tarfile
import os
import matplotlib.pyplot as plt
from preprocessDefinition import preprocess

# Constants Assumed
NUM_CLASSES = 358
BATCH_SIZE = 32
IMAGE_SIZE = 299
EPOCHS_PHASE1 = 5
EPOCHS_PHASE2 = 5
TRAIN_SIZE = 7160   # 20 images x 358 classes
VAL_SIZE = 3580     # 10 images x 358 classes

# Parsing tfrecord
def parse_examples(serialized_examples):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'birdType': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_example(serialized_examples, feature_description)
    image = tf.io.decode_jpeg(example['image'], channels=3)
    label = example['birdType']
    return image, label

# Creating function that loads Dataset provided
def load_dataset(tfrecord_path, repeat=False):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_examples, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = load_dataset('birds-20-eachOf-358.tfrecords', repeat=True)
val_dataset = load_dataset('birds-10-eachOf-358.tfrecords')

# Using Pretrained model : Xception
base_model = tf.keras.applications.Xception(
    include_top=False,
    weights='imagenet',
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
)
base_model.trainable = False

# Classification Head
inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

# Metrics to evaluate
top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5")
top10 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name="top10")
top20 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=20, name="top20")

# Compiler for Phase 1 - Freezing Base Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', top5, top10, top20]
)

# Callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('best_birder.keras', save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
]

# Training Phase 1
history1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_PHASE1,
    steps_per_epoch=TRAIN_SIZE // BATCH_SIZE,
    validation_steps=VAL_SIZE // BATCH_SIZE,
    callbacks=callbacks
)

# Phase 2 - Unfreezing last 30 layers and fine tuning them due to less computational power
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', top5, top10, top20]
)

history2 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    initial_epoch=EPOCHS_PHASE1,
    epochs=EPOCHS_PHASE1 + EPOCHS_PHASE2,
    steps_per_epoch=TRAIN_SIZE // BATCH_SIZE,
    validation_steps=VAL_SIZE // BATCH_SIZE,
    callbacks=callbacks
)

# Saving final models
model.save('birderModel.keras')
with tarfile.open('birderModel.tgz', 'w:gz') as tar:
    tar.add('birderModel.keras', arcname='birderModel.keras')

# Evaluating results for comparison
results = model.evaluate(val_dataset, steps=VAL_SIZE // BATCH_SIZE)
top1, top5, top10, top20 = results[1], results[2], results[3], results[4]
average = (top1 + top5 + top10 + top20) / 4

print(f"Top-1 Accuracy:  {top1 * 100:.2f}%")
print(f"Top-5 Accuracy:  {top5 * 100:.2f}%")
print(f"Top-10 Accuracy: {top10 * 100:.2f}%")
print(f"Top-20 Accuracy: {top20 * 100:.2f}%")
print(f"Average Top-K Accuracy: {average * 100:.2f}%")

# Plotting Training vs Validation Accuracy and Loss
history = history1.history
for k, v in history2.history.items():
    if k in history:
        history[k] += v
    else:
        history[k] = v

epochs_range = range(1, len(history['accuracy']) + 1)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, history['accuracy'], label='Train Acc')
plt.plot(epochs_range, history['val_accuracy'], label='Val Acc')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history['loss'], label='Train Loss')
plt.plot(epochs_range, history['val_loss'], label='Val Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

