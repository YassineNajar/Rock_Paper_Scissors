import cv2
import mediapipe as mp
import random
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Define Rock, Paper, Scissors moves
gestures = {0: "Rock", 1: "Paper", 2: "Scissors"}


# Function to classify hand gesture
def classify_hand(landmarks):
    thumb_tip = landmarks.landmark[4]
    index_finger_tip = landmarks.landmark[8]
    middle_finger_tip = landmarks.landmark[12]
    ring_finger_tip = landmarks.landmark[16]
    pinky_tip = landmarks.landmark[20]

    def distance(a, b):
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

    thumb_index_dist = distance(thumb_tip, index_finger_tip)
    index_middle_dist = distance(index_finger_tip, middle_finger_tip)
    middle_ring_dist = distance(middle_finger_tip, ring_finger_tip)
    ring_pinky_dist = distance(ring_finger_tip, pinky_tip)
    thumb_pinky_dist = distance(thumb_tip, pinky_tip)

    # Heuristics for gesture classification
    if thumb_index_dist < 0.1 and index_middle_dist < 0.1 and middle_ring_dist < 0.1 and ring_pinky_dist < 0.1:
        return "Rock"
    elif thumb_index_dist > 0.1 and index_middle_dist > 0.1 and middle_ring_dist > 0.1 and ring_pinky_dist > 0.1:
        return "Paper"
    elif index_middle_dist < 0.3 and middle_ring_dist > 0.23 and ring_pinky_dist < 0.15:
        return "Scissors"
    return None


# Function to determine the winner
def determine_winner(user_move, system_move):
    if user_move == system_move:
        return "Draw"
    if (user_move == "Rock" and system_move == "Scissors") or \
            (user_move == "Paper" and system_move == "Rock") or \
            (user_move == "Scissors" and system_move == "Paper"):
        return "User Wins"
    else:
        return "System Wins"


# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Start the countdown timer for 3 seconds
    start_time = time.time()
    countdown = 3

    user_move = None

    while time.time() - start_time < countdown:
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display
        # Convert the BGR image to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # Process the image and find hands
        results = hands.process(image)

        # Convert the image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Classify the hand gesture
                user_move = classify_hand(hand_landmarks)
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the countdown
        cv2.putText(image, f'Time left: {int(countdown - (time.time() - start_time))}',
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Rock Paper Scissors', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            hands.close()
            exit()

    # Generate system move
    system_move = gestures[random.randint(0, 2)]

    # Determine the winner
    result = "No gesture detected"
    if user_move:
        result = determine_winner(user_move, system_move)

    # Display the result
    success, image = cap.read()
    image = cv2.flip(image, 1)
    cv2.putText(image, f'User: {user_move}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(image, f'System: {system_move}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image, f'Result: {result}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('Rock Paper Scissors', image)

    # Wait for a while to display the result before starting the next round
    if cv2.waitKey(3000) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
hands.close()
