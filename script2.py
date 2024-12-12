import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical

# Mediapipe init
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Paths and settings
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['menu-item1', 'menu-item2', 'menu-item3', 'exit'])
no_sequences = 30
sequence_length = 30
label_map = {label: num for num, label in enumerate(actions)}

# Create folders for data storage
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except FileExistsError:
            pass

# Mediapipe detection func
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Draw styled landmarks
def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )

# Extract right-hand keypoints
def extract_right_hand_keypoints(results):
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)
    return rh

# Data collection
def collect_data():
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            for sequence in range(no_sequences):
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Unable to read from the webcam.")
                        continue

                    # Perform Mediapipe detection
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    # Display collection status
                    if frame_num == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
                        cv2.putText(image, f'Collecting frames for {action}, Video {sequence}', (50, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(image, f'Collecting frames for {action}, Video {sequence}', (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                    # Extract keypoints and save
                    keypoints = extract_right_hand_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                    np.save(npy_path, keypoints)

                    # Display the frame
                    cv2.imshow('OpenCV Feed', image)

                    # Break when htting on 'q'
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
    cap.release()
    cv2.destroyAllWindows()

# Prepare data for training
def prepare_data():
    sequences, labels = [], []
    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                res = np.load(npy_path)
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    X = np.array(sequences)
    Y = to_categorical(labels).astype(int)
    return train_test_split(X, Y, test_size=0.05)

# Build and train the model
def train_model(X_train, Y_train):
    log_dir = os.path.join('logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(X_train, Y_train, epochs=4000, callbacks=[tb_callback])
    model.save('action.h5')

# Real-time testing
def real_time_test():
    model = load_model('action.h5')  # Load the trained model
    sequence = []
    threshold = 0.6  # Confidence threshold for predictions

    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read from the webcam.")
                break

            # Perform Mediapipe detection
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            # Extract keypoints
            keypoints = extract_right_hand_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]  # Use the last 30 frames for predictions

            # Predict action
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                if res[np.argmax(res)] > threshold:
                    action = actions[np.argmax(res)]
                    print(f"Action: {action} (Confidence: {res[np.argmax(res)]:.2f})")
                    cv2.putText(image, f'Action: {action}', (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, 'Action: None', (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Display Webcam Feed
            cv2.imshow('Real-Time Testing', image)

            # Exit on 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Main script
if __name__ == "__main__":
    mode = input("Choose mode (train/test): ").strip().lower() ## interactive menu to select between train or test
    if mode == "train":
        print("Preparing data for training...")
        X_train, X_test, Y_train, Y_test = prepare_data()
        print("Training the model...")
        train_model(X_train, Y_train)
        print("Training completed.")
    elif mode == "test":
        print("Starting real-time testing...")
        real_time_test()
    else:
        print("Invalid mode. Please choose 'train' or 'test'.")
