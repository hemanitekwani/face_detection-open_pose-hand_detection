import cv2
import mediapipe as mp
from mediapipe.python.solutions import drawing_utils

# Step 1: Initialize Mediapipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Step 2: Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 to capture video from the default camera

while True:
    # Step 3: Read frame from video capture
    ret, frame = cap.read()

    if not ret:
        break

    # Step 4: Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Step 5: Process the image with Mediapipe Pose
    results = pose.process(image)

    # Step 6: Draw the pose landmarks on the image
    if results.pose_landmarks:
        drawing_utils.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Step 7: Display the image
    cv2.imshow('OpenPose Detection', image)

    # Step 8: Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 9: Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
