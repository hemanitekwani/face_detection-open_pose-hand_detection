import cv2
import mediapipe as mp
from mediapipe.python.solutions import drawing_utils


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)  # Use 0 to capture video from the default camera

while True:
 
    ret, frame = cap.read()

    if not ret:
        break

  
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    results = pose.process(image)

    
    if results.pose_landmarks:
        drawing_utils.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

  
    cv2.imshow('OpenPose Detection', image)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
