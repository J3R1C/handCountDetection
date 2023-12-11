import mediapipe as mp
import cv2
import csv

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open a CSV file to write landmarks
csv_file = open('hand_landmarks.csv', mode='w', newline='')
csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

header = ['Number', 'Thumb', 'Index', 'Middle', 'Ring', 'Little']
csv_writer.writerow(header)

csv_file = open('xyz_landmarks.csv', mode='w', newline='')
csv_writer1 = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

header = ['Number']
for i in range(21):  # Assuming 21 landmarks
    header.extend([f'Landmark_{i}_X', f'Landmark_{i}_Y', f'Landmark_{i}_Z'])
csv_writer1.writerow(header)

# Capture video from webcam
cap = cv2.VideoCapture(0)

def recognize_number(landmarks):
    # Counting numbers 1 to 5 based on hand landmarks
    # Define the conditions to recognize numbers 1 to 5 based on hand gestures
    
    count = 0  # Initialize count
    
    # Right or Left Thumb up
    if landmarks[4].x < landmarks[3].x:  # Checking if thumb is up (comparing X-coordinates)
        count += 1
    # Index, Middle, Ring, Little fingers up
    if landmarks[8].y < landmarks[6].y:  # Checking if index finger is raised (comparing Y-coordinates)
        count += 1
    if landmarks[12].y < landmarks[10].y:  # Checking if middle finger is raised (comparing Y-coordinates)
        count += 1
    if landmarks[16].y < landmarks[14].y:  # Checking if ring finger is raised (comparing Y-coordinates)
        count += 1
    if landmarks[20].y < landmarks[18].y:  # Checking if little finger is raised (comparing Y-coordinates)
        count += 1
    
    return count  # Return the count

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Get hand landmarks
    results = hands.process(image)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:  

            landmarks = hand_landmarks.landmark
            csv_writer1.writerow([landmark.x for landmark in landmarks] + [landmark.y for landmark in landmarks] + [landmark.z for landmark in landmarks])     
            
            # Recognize number gesture
            number = recognize_number(hand_landmarks.landmark)
            landmark_data = [number]
            landmark_data.extend([1 if number == i else 0 for i in range(1, 6)])
            csv_writer.writerow(landmark_data)
            
            if number != -1:
                cv2.putText(frame, f"Number: {number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
            # Draw hand landmarks on the image
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    cv2.imshow('Counting Numbers', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close CSV file
cap.release()
csv_file.close()
cv2.destroyAllWindows()

