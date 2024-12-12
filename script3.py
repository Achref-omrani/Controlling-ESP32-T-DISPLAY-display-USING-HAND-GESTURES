import os
import cv2
import numpy as np
import mediapipe as mp

# Mediapipe initialization
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Paths and settings
DATA_PATH = os.path.join('MP_Data')
actions = np.array(['menu-item1', 'menu-item2', 'menu-item3', 'Exit'])  # Modify actions if needed
no_sequences = 30  # Number of sequences for each action *****change as much requiured
sequence_length = 30  # Number of frames in each sequence  *****change as much requiured

# Create folders for data storage
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except FileExistsError:
            pass

# Mediapipe detection function
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
    cap = cv2.VideoCapture(0)  # Start the webcam
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            print(f"Collecting data for action: {action}")
            for sequence in range(no_sequences):
                print(f"Sequence {sequence + 1}/{no_sequences}")
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
                        cv2.putText(image, f'STARTING {action.upper()} COLLECTION', (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(2000)  # Wait for 2 seconds before starting

                    # Extract keypoints and save
                    keypoints = extract_right_hand_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                    np.save(npy_path, keypoints)

                    # Display the frame
                    cv2.putText(image, f'Collecting {action}: Sequence {sequence + 1}, Frame {frame_num + 1}',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)

                    # Break on 'q'
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        print("Exiting data collection...")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
    cap.release()
    cv2.destroyAllWindows()

# Main script
if __name__ == "__main__":
    print("Starting data collection...")
    collect_data()
    print("Data collection completed.")
