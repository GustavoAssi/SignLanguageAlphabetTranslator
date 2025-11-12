import cv2
import mediapipe as mp
import numpy as np
import os
import string


IMAGE_SIZE = 224
DATA_DIR = "data"


def create_data_dirs():
    """
    Cria diretório principal de dados, se ainda não existir.
    """
    
    os.makedirs(DATA_DIR, exist_ok=True)
    print("[INFO] Diretório de dados criado/verificado.")


def init_mediapipe():
    """
    Inicializa o detector de mãos do MediaPipe.
    """

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    return hands, mp_hands, mp_drawing


def capture_webcam():
    """
    Inicia a captura da webcam.
    """

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Erro ao acessar a webcam.")
    print("[INFO] Webcam iniciada com sucesso.")
    return cap


def process_frame(frame, hands, mp_hands, mp_drawing):
    """
    Processa o frame e retorna o recorte da mão.
    """

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    hand_img = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # desenha landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # calcula a bounding box da mão
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            xmin = int(min(x_coords) * w) - 20
            ymin = int(min(y_coords) * h) - 20
            xmax = int(max(x_coords) * w) + 20
            ymax = int(max(y_coords) * h) + 20

            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w, xmax), min(h, ymax)

            # recorta e redimensiona
            hand_img = frame[ymin:ymax, xmin:xmax]
            if hand_img.size != 0:
                hand_img = cv2.resize(hand_img, (IMAGE_SIZE, IMAGE_SIZE))
            break
    return frame, hand_img


def save_image(hand_img, letter, count):
    """
    Salva a imagem recortada.
    """

    letter_dir = os.path.join(DATA_DIR, letter)
    os.makedirs(letter_dir, exist_ok=True)
    save_path = os.path.join(letter_dir, f"{count:04d}.jpg")
    cv2.imwrite(save_path, hand_img)
    print(f"[OK] Imagem salva: {save_path}")


def main():
    create_data_dirs()
    hands, mp_hands, mp_drawing = init_mediapipe()
    cap = capture_webcam()

    # pergunta qual letra será coletada
    current_letter = input("Digite a letra que deseja coletar (A-Z): ").strip().upper()
    if current_letter not in string.ascii_uppercase:
        print("[ERRO] Letra inválida. Use apenas A-Z.")
        return

    # cria janelas uma única vez
    cv2.namedWindow("Coleta de dados")
    cv2.namedWindow("Recorte da mão")

    # conta imagens já existentes
    img_count = len(os.listdir(os.path.join(DATA_DIR, current_letter))) if os.path.exists(
        os.path.join(DATA_DIR, current_letter)
    ) else 0
    print(f"[INFO] Coletando imagens para '{current_letter}' (iniciando em {img_count})...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, hand_img = process_frame(frame, hands, mp_hands, mp_drawing)

        cv2.putText(frame, f"Letra: {current_letter}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # atualiza janelas já criadas
        cv2.imshow("Coleta de dados", frame)

        if hand_img is not None and hand_img.size != 0:
            cv2.imshow("Recorte da mão", hand_img)

        key = cv2.waitKey(1) & 0xFF

        # salvar imagem
        if key == ord('s') and hand_img is not None and hand_img.size != 0:
            save_image(hand_img, current_letter, img_count)
            img_count += 1

        # sair (ESC)
        elif key == 27:
            print(f"\n[INFO] Coleta finalizada. Total de imagens: {img_count}")
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
