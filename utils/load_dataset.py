import numpy as np
import math
import tensorflow as tf
from keras import (
    layers,
    models,
    optimizers,
    utils,
    callbacks,
    metrics,
    losses,
    activations,
)

def load_flowers_dataset(image_size: int = 64, dataset_repetitions: int = 5, batch_size: int = 64) -> tf.data.Dataset:

    train_data = utils.image_dataset_from_directory(
        "/app/data/pytorch-challange-flower-dataset/dataset",
        labels=None,
        image_size=(image_size, image_size),
        batch_size=None,
        shuffle=True,
        seed=42,
        interpolation="bilinear",
        )

    train = train_data.map(lambda x: preprocess(x))
    train = train.repeat(dataset_repetitions)
    train = train.batch(batch_size, drop_remainder=True)

    return train

# Preprocess the data
def preprocess(img: tf.Tensor) -> tf.Tensor:
    img = tf.cast(img, "float32") / 255.0
    return img