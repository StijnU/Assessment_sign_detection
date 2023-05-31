import pickle
import json
import numpy as np
import mediapipe as mp
import cv2
import sys
from sklearn.linear_model import LogisticRegression


def detect_hand_landmarks(image):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # Load the MediaPipe Hands model
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        # Convert the image to RGB format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image to detect hand landmarks
        results = hands.process(image_rgb)

        # Check if any hands were detected
        if results.multi_hand_landmarks:
            # Iterate through each detected hand
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the landmarks on the image
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract the landmarks for each finger
                for finger_id in range(5):
                    finger_landmarks = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST + (finger_id * 4):mp_hands.HandLandmark.WRIST + (finger_id * 4) + 4]

    # Return the image with landmarks drawn
    return image, results

def preprocess_landmarks(landmarks):
    return np.array([[point.x, point.y, point.z]for point in landmarks.multi_hand_landmarks[0].landmark]).flatten()

model_path = sys.argv[1]
img_json_path = sys.argv[2]


with open(model_path, 'rb') as read_file:
    model = pickle.load(read_file)


with open(img_json_path, 'r') as read_file:
    img = json.load(read_file)

img_data = np.array(img['data'],  dtype=np.uint8)



signs = ['open-hand', 'rock-on', 'peace', 'surf']

_, landmarks = detect_hand_landmarks(img_data)

if landmarks.multi_hand_landmarks:
    prep_landmarks = preprocess_landmarks(landmarks)
    prediction = model.predict_proba([prep_landmarks])
    text = signs[np.argmax(prediction)]
print('Prediction: ' +  text)