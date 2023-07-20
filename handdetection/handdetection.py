import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)


image = cv2.imread('hand_image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = hands.process(image_rgb)


if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        for landmark in hand_landmarks.landmark:
           
            x = int(landmark.x * image.shape[1])
            y = int(landmark.y * image.shape[0])
            z = landmark.z  
           
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

cv2.imshow('Hand Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
