import cv2
import math
import time
import mediapipe as mp
from deepface import DeepFace
from xarm.wrapper import XArmAPI


# Configure the xArm
arm = XArmAPI('192.168.1.201')  # Use Robot's IP
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

# Initial Position
x_pos = 200
arm.set_position(246.6, -30.6, 236, 176.6, -2.2, -1.5, speed=50, wait=True)

def increment_dictionary_value(dictionary, key):
  """Increments the value associated with a given key in a dictionary.

  Args:
    dictionary: A dictionary with integer values.
    key: The key whose value should be incremented.
  """
  if key in dictionary:
    dictionary[key] += 1

def get_max_value_key(dictionary):
  """Returns the key with the highest value from a dictionary.

  Args:
    dictionary: A dictionary with integer values.

  Returns:
    The key associated with the highest integer value in the dictionary.
  """
  if not dictionary:
    return None # Handle empty dictionary case

  max_value = None
  max_key = None

  for key, value in dictionary.items():
    if max_value is None or value > max_value:
      max_value = value
      max_key = key

  return max_key


# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

i = 0
detected_emotion = ""
emotion = ""
wait_seconds = 3
wait_detection_seconds = 2.9 # Must be less than wait seconds

max_i = 10

stop_time = time.perf_counter()

k = 0
text = "Detecting emotion"
color_tuple = (0, 255, 0)

start_detection_time = time.perf_counter()

emotions_counter_dict = {"happy" : 0, "sad" : 0, "angry" : 0, "neutral" : 0, "surprise" : 0}

while True:

    if stop_time + wait_seconds < time.perf_counter():
        k = 0
        stop_time = time.perf_counter()
        start_detection_time = time.perf_counter()
        text = "Detecting emotion"
        color_tuple = (0, 255, 0)

    # Capture frame-by-frame
    ret, frame = cap.read()

    elapsed_time = time.perf_counter() - stop_time

    time_text = text + f" {elapsed_time:.1f} seconds."

    cv2.putText(frame,time_text,(20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_tuple, 2)

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    j = 0

    if k == 0:

        for (x, y, w, h) in faces:

            # Extract the face ROI (Region of Interest)
            face_roi = rgb_frame[y:y + h, x:x + w]


            # Perform emotion analysis on the face ROI
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            # Determine the dominant emotion
            emotion = result[0]['dominant_emotion']

            increment_dictionary_value(emotions_counter_dict, emotion)

            # Draw rectangle around face and label with predicted emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        elapsed_detection_time = time.perf_counter() - start_detection_time

        if elapsed_detection_time >= wait_detection_seconds :

            max_emotion_key = get_max_value_key(emotions_counter_dict)
            print(max_emotion_key)

            # Send instructions to Robot according to max_emotion_key

            if max_emotion_key == 'happy':
              arm.tcp_acc = 2000
              arm.tcp_speed = 200
              arm.angle_acc = 2000
              arm.angle_speed = 200
              arm.set_servo_angle(angle=[-7.0, -35.1, -7.3, -4.9, 40.1, -1.7], speed=arm.angle_speed, mvacc=arm.angle_acc, wait=False, radius=0.0)
              arm.set_position(*[246.6, -30.6, 198.9, 176.6, -2.2, -1.5], speed=arm.tcp_speed, mvacc=arm.tcp_acc, radius=-1.0, wait=True)
              arm.move_circle([271.6, -5.6, 198.9, 176.6, -2.2, -1.5], [271.6, -55.6, 198.9, 176.6, -2.2, -1.5], float(180) / 360 * 100, speed=arm.tcp_speed, mvacc=arm.tcp_acc, wait=True)
              arm.set_servo_angle(angle=[-7.0, -35.1, -7.3, -4.9, 40.1, -1.7], speed=arm.angle_speed, mvacc=arm.angle_acc, wait=False, radius=0.0)

            if max_emotion_key == 'sad':
              arm.tcp_acc = 2000
              arm.tcp_speed = 200
              arm.angle_acc = 2000
              arm.angle_speed = 200
              arm.set_servo_angle(angle=[-7.0, -35.1, -7.3, -4.9, 40.1, -1.7], speed=arm.angle_speed, mvacc=arm.angle_acc, wait=False, radius=0.0)
              arm.set_position(*[246.6, -30.6, 198.9, 176.6, -2.2, -1.5], speed=arm.tcp_speed, mvacc=arm.tcp_acc, radius=-1.0, wait=True)
              arm.move_circle([271.6, -55.6, 198.9, 176.6, -2.2, -1.5], [271.6, -5.6, 198.9, 176.6, -2.2, -1.5], float(180) / 360 * 100, speed=arm.tcp_speed, mvacc=arm.tcp_acc, wait=True)
              arm.set_servo_angle(angle=[-7.0, -35.1, -7.3, -4.9, 40.1, -1.7], speed=arm.angle_speed, mvacc=arm.angle_acc, wait=False, radius=0.0)

            if max_emotion_key == 'neutral':
              arm.tcp_speed = 200
              arm.tcp_acc = 2000
              arm.angle_acc = 2000
              arm.angle_speed = 200
              arm.set_servo_angle(angle=[-7.0, -35.1, -7.3, -4.9, 40.1, -1.7], speed=arm.angle_speed, mvacc=arm.angle_acc, wait=False, radius=0.0)
              arm.set_position(*[246.6, -30.6, 198.9, 176.6, -2.2, -1.5], speed=arm.tcp_speed, mvacc=arm.tcp_acc, radius=-1.0, wait=True)
              arm.set_position(*[306.6, -30.6, 198.9, 176.6, -2.2, -1.5], speed=arm.tcp_speed, mvacc=arm.tcp_acc, radius=-1.0, wait=True)
              arm.set_servo_angle(angle=[-7.0, -35.1, -7.3, -4.9, 40.1, -1.7], speed=arm.angle_speed, mvacc=arm.angle_acc, wait=False, radius=0.0)

            if max_emotion_key == 'surprise':
                arm.tcp_acc = 2000
                arm.tcp_speed = 200
                arm.angle_acc = 2000
                arm.angle_speed = 200
                arm.set_servo_angle(angle=[-7.0, -35.1, -7.3, -4.9, 40.1, -1.7], speed=arm.angle_speed, mvacc=arm.angle_acc, wait=False, radius=0.0)
                arm.set_position(*[246.6, -30.6, 198.9, 176.6, -2.2, -1.5], speed=arm.tcp_speed, mvacc=arm.tcp_acc, radius=-1.0, wait=True)
                arm.move_circle([271.6, -55.6, 198.9, 176.6, -2.2, -1.5], [271.6, -5.6, 198.9, 176.6, -2.2, -1.5], float(360) / 360 * 100, speed=arm.tcp_speed, mvacc=arm.tcp_acc, wait=True)
                arm.set_servo_angle(angle=[-7.0, -35.1, -7.3, -4.9, 40.1, -1.7], speed=arm.angle_speed, mvacc=arm.angle_acc, wait=False, radius=0.0)



            k = 1
            # i = 0
            stop_time = time.perf_counter()
            text = "Detection Paused"
            color_tuple = (0, 0, 255)
            emotions_counter_dict = {"happy" : 0, "sad" : 0, "angry" : 0, "neutral" : 0, "surprise" : 5}

        detected_emotion = emotion

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
