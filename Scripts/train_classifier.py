import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



BASE_DIR = os.path.dirname(os.path.abspath(__file__))       
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))    

DATASET_PATH = os.path.join(ROOT_DIR, "dataset")
MODEL_NAME = os.path.join(ROOT_DIR, "model_sign_language.h5")
CLASS_MAPPING_PATH = os.path.join(ROOT_DIR, "class_mapping.json")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50


def load_data():
    """
    Carrega treino, validação e teste separadamente,
    garantindo reprodutibilidade e uma avaliação mais precisa.
    """

    train_datagen = ImageDataGenerator(
        rescale=1/255,
        rotation_range=12,
        zoom_range=0.10,
        width_shift_range=0.10,
        height_shift_range=0.10,
        shear_range=0.1,
        brightness_range=[0.8, 1.2],
        horizontal_flip=False,  
        validation_split=0.2
    )

    
    train_gen = train_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, "train"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    # Validação (usada durante o treinamento)
    val_gen = train_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, "train"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        class_mode="categorical",
        subset="validation",
        shuffle=True
    )

    
    test_datagen = ImageDataGenerator(rescale=1/255)

    test_gen = test_datagen.flow_from_directory(
        os.path.join(DATASET_PATH, "test"),
        target_size=IMAGE_SIZE,
        batch_size=1,
        color_mode="rgb",
        class_mode="categorical",
        shuffle=False    
    )

    print("[INFO] Classes detectadas:", train_gen.class_indices)
    return train_gen, val_gen, test_gen



def build_cnn_model(num_classes):
    """
    CNN mais robusta, inspirada em VGG-like, com regularização.
    """

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', padding="same", input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        Conv2D(32, (3,3), activation='relu', padding="same"),
        MaxPooling2D(2,2),
        Dropout(0.25),

        Conv2D(64, (3,3), activation='relu', padding="same"),
        Conv2D(64, (3,3), activation='relu', padding="same"),
        MaxPooling2D(2,2),
        Dropout(0.30),

        Conv2D(128, (3,3), activation='relu', padding="same"),
        Conv2D(128, (3,3), activation='relu', padding="same"),
        MaxPooling2D(2,2),
        Dropout(0.40),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model



def train_model(model, train_gen, val_gen):

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ModelCheckpoint("model_sign_language.h5", save_best_only=True)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    return history


def save_model(model, filename=MODEL_NAME):
    model.save(filename)
    print(f"[INFO] Modelo salvo em: {filename}")


def save_class_mapping(train_gen, filename=CLASS_MAPPING_PATH):
    with open(filename, "w") as f:
        json.dump(train_gen.class_indices, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Mapeamento salvo em: {filename}")


def evaluate_model(model, test_gen, idx_to_class):

    preds = model.predict(test_gen, verbose=1)
    pred_labels = np.argmax(preds, axis=1)
    true_labels = test_gen.classes

    
    cm = confusion_matrix(true_labels, pred_labels)


    pd.DataFrame(
        cm,
        index=list(idx_to_class.values()),
        columns=list(idx_to_class.values())
    ).to_csv("confusion_matrix.csv")

    print("\n[INFO] Matriz de confusão salva como confusion_matrix.csv")

    
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(idx_to_class.values()),
                yticklabels=list(idx_to_class.values()))
    plt.xlabel("Predição")
    plt.ylabel("Verdadeiro")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    print("[INFO] Matriz de confusão salva como confusion_matrix.png\n")

    
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=list(idx_to_class.values()),
        output_dict=False
    )

    print(report)


def main():
    train_gen, val_gen, test_gen = load_data()
    num_classes = len(train_gen.class_indices)

    model = build_cnn_model(num_classes)
    history = train_model(model, train_gen, val_gen)

    save_class_mapping(train_gen)
    save_model(model)

    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    evaluate_model(model, test_gen, idx_to_class)


if __name__ == "__main__":
    main()
