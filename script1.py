import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import serial
import time

# Mediapipe Hands init
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Actions and sequence settings
actions = np.array(['menu-item1', 'menu-item2', 'menu-item3', 'menu-item4'])
sequence_length = 30
prediction_history = []  # History for smoothing predictions
history_limit = 10       # Number of frames to consider for smoothing

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
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
            )

# Extract right-hand keypoints
def extract_right_hand_keypoints(results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            return np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
    return np.zeros(21 * 3)

# Real-time testing with ESP32 communication
def real_time_test_with_esp32(port="COM5", baudrate=115200):
    global prediction_history  # Declare the global variable
    model = load_model('action.h5')  # Load the trained model
    sequence = []
    threshold = 0.6  # Confidence threshold for predictions
    last_action = None  # Track the last detected action
    last_sent_time = time.time()  # Timestamp of the last sent action
    debounce_interval = 1  # Minimum interval (in seconds) between actions

    # Init serial communication
    try:
        esp32 = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # Wait for the connection to establish
        print(f"Connected to ESP32 on {port} at {baudrate} baudrate.")
    except Exception as e:
        print(f"Failed to connect to ESP32: {e}")
        return

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read from the webcam.")
                break

            # Perform Mediapipe detection
            image, results = mediapipe_detection(frame, hands)
            draw_styled_landmarks(image, results)

            # Extract keypoints
            keypoints = extract_right_hand_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]  # Use the last 30 frames for predictions

            # Predict action
            if len(sequence) == 30 and not np.all(sequence[-1] == 0):  # Ensure last keypoints are not all zeros
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                prediction_history.append(res)
                prediction_history = prediction_history[-history_limit:]  # Keep history within the limit

                # smoothing
                smoothed_res = np.mean(prediction_history, axis=0)
                if smoothed_res[np.argmax(smoothed_res)] > threshold:
                    action = actions[np.argmax(smoothed_res)]
                    current_time = time.time()

                    # Only send action if it has changed and debounce interval has passed
                    if action != last_action and (current_time - last_sent_time > debounce_interval):
                        print(f"Action: {action} (Confidence: {smoothed_res[np.argmax(smoothed_res)]:.2f})")
                        cv2.putText(image, f'Action: {action}', (10, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        # Send action to ESP32
                        esp32.write(f"{action}\n".encode('utf-8'))
                        last_action = action
                        last_sent_time = current_time
                else:
                    cv2.putText(image, 'Action: None', (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                # Reset last_action if no hand is detected
                last_action = None
                cv2.putText(image, 'Action: None', (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Display Webcam Feed
            cv2.imshow('Real-Time Testing', image)

            # Exit on 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    esp32.close()


# Main script
if __name__ == "__main__":
    print("Starting real-time testing with ESP32 communication...")
    real_time_test_with_esp32(port="COM5")
