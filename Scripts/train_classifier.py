import os
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping


BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # Pasta Scripts/
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))    # Pasta raiz do projeto

DATASET_PATH = os.path.join(ROOT_DIR, "dataset")
MODEL_NAME = os.path.join(ROOT_DIR, "model_sign_language.h5")
CLASS_MAPPING_PATH = os.path.join(ROOT_DIR, "class_mapping.json")


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30


def load_data():
    """
    Carrega e prepara os geradores de treino e teste.
    """

    datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        os.path.join(DATASET_PATH, "train"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    test_gen = datagen.flow_from_directory(
        os.path.join(DATASET_PATH, "train"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        class_mode="categorical",
        subset="validation",
        shuffle=True
    )

    print(f"[INFO] Classes detectadas: {train_gen.class_indices}")
    return train_gen, test_gen


def build_cnn_model(num_classes):
    """
    Cria uma CNN simples para classificação de imagens.
    """

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model, train_gen, test_gen):
    """
    Treina o modelo com EarlyStopping.
    """

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=test_gen,
        callbacks=[early_stop],
        verbose=1
    )
    return history


def save_model(model, filename=MODEL_NAME):
    """
    Salva o modelo treinado.
    """

    model.save(filename)
    print(f"[INFO] Modelo salvo em: {filename}")


def save_class_mapping(train_generator, filename=CLASS_MAPPING_PATH):
    """
    Salva o mapeamento class_name -> index em JSON.
    """

    class_indices = train_generator.class_indices
    with open(filename, "w") as f:
        json.dump(class_indices, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Mapeamento de classes salvo em: {filename}")



def main():
    train_gen, test_gen = load_data()
    num_classes = len(train_gen.class_indices)
    model = build_cnn_model(num_classes)
    history = train_model(model, train_gen, test_gen)
    save_class_mapping(train_gen)
    save_model(model)


if __name__ == "__main__":
    main()
