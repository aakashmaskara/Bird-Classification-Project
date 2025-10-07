import tensorflow as tf

def preprocess(image, label):
    image = tf.image.resize_with_pad(image, 299, 299)
    image = tf.keras.applications.xception.preprocess_input(image)
    return image, label