import os
import shutil
import random


def reset_dataset_dir(dest_dir="dataset"):
    """
    Remove o diretório do dataset, se existir, e cria novamente.
    """

    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
        print(f"[INFO] Diretório '{dest_dir}/' removido.")
    os.makedirs(dest_dir, exist_ok=True)
    print(f"[INFO] Diretório '{dest_dir}/' recriado.")


def create_dir_structure(base_dir="dataset"):
    """
    Cria estrutura de diretórios para treino e teste.
    """

    for split in ["train", "test"]:
        split_path = os.path.join(base_dir, split)
        os.makedirs(split_path, exist_ok=True)
    print("[INFO] Estrutura de diretórios criada/verificada.")


def split_and_copy_data(source_dir="data", dest_dir="dataset", train_ratio=0.8):
    """
    Divide as imagens de cada classe e copia para treino/teste.
    """
    
    letters = sorted(os.listdir(source_dir))
    total_images = 0

    for letter in letters:
        letter_path = os.path.join(source_dir, letter)
        if not os.path.isdir(letter_path):
            continue

        images = os.listdir(letter_path)
        random.shuffle(images)

        split_idx = int(len(images) * train_ratio)
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]

        # cria pastas de destino
        train_dir = os.path.join(dest_dir, "train", letter)
        test_dir = os.path.join(dest_dir, "test", letter)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # copia arquivos
        for img in train_imgs:
            shutil.copy(os.path.join(letter_path, img), os.path.join(train_dir, img))
        for img in test_imgs:
            shutil.copy(os.path.join(letter_path, img), os.path.join(test_dir, img))

        print(f"[OK] Letra '{letter}': {len(train_imgs)} treino / {len(test_imgs)} teste")
        total_images += len(images)

    print(f"\n[INFO] Total de imagens processadas: {total_images}")


def main():
    reset_dataset_dir("dataset")
    create_dir_structure("dataset")
    split_and_copy_data("data", "dataset", train_ratio=0.8)


if __name__ == "__main__":
    main()
