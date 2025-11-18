from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Dropout,
    Flatten,
)


def build_model(input_shape, num_classes):

    model = Sequential(
        [
            # Batch 1
            Conv2D(
                32, (3, 3), input_shape=input_shape, padding="same", activation="relu"
            ),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            # Batch 2
            Conv2D(64, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            # Batch 3
            Conv2D(128, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            # Batch 4
            Conv2D(256, (3, 3), padding="same", activation="relu"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model
