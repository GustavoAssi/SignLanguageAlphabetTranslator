import os
import cv2
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp


BASE_DIR = os.path.dirname(os.path.abspath(__file__))       
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))    

MODEL_PATH = os.path.join(ROOT_DIR, "model_sign_language.h5")
CLASS_MAPPING_PATH = os.path.join(ROOT_DIR, "class_mapping.json")


def load_class_mapping(mapping_path=CLASS_MAPPING_PATH):
    """
    Carrega e inverte o mapeamento de classes.
    """

    with open(mapping_path, "r") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    print(f"[INFO] Mapeamento de classes carregado: {idx_to_class}")
    return idx_to_class


def preprocess_frame(frame, hand_landmarks, image_size=(224, 224)):
    """
    Recorta e pré-processa a região da mão detectada.
    """

    h, w, _ = frame.shape
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]

    x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
    y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

    offset = 20
    x_min, x_max = max(0, x_min - offset), min(w, x_max + offset)
    y_min, y_max = max(0, y_min - offset), min(h, y_max + offset)

    hand_img = frame[y_min:y_max, x_min:x_max]
    hand_img = cv2.resize(hand_img, image_size)
    hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
    hand_img = hand_img / 255.0
    return np.expand_dims(hand_img, axis=0)


def predict_letter(model, frame, hand_landmarks, idx_to_class):
    """
    Realiza predição e retorna letra + percentual de confiança.
    """

    img = preprocess_frame(frame, hand_landmarks)
    preds = model.predict(img, verbose=0)[0]
    pred_idx = np.argmax(preds)
    pred_class = idx_to_class[pred_idx]
    confidence = preds[pred_idx] * 100
    return pred_class, confidence


def main():
    
    print("[INFO] Carregando modelo e mapeamento de classes...")
    model = load_model(MODEL_PATH)
    idx_to_class = load_class_mapping(CLASS_MAPPING_PATH)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERRO] Não foi possível acessar a webcam.")
        return

    print("[INFO] Pressione ESC para sair.")
    last_pred = None
    stable_count = 0
    last_conf = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERRO] Falha na captura de frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                pred, conf = predict_letter(model, frame, hand_landmarks, idx_to_class)

                if pred == last_pred:
                    stable_count += 1
                else:
                    stable_count = 0
                    last_pred = pred
                    last_conf = conf

                if stable_count > 3:
                    cv2.putText(frame, f"{pred} ({last_conf:.1f}%)",
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                                2, (0, 255, 0), 3)

        cv2.imshow("Classificador de Libras - CNN", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
