import math
import cv2
import mediapipe as mp
import numpy as np
from xarm.wrapper import XArmAPI
from deepface import DeepFace


# Configure xArm
arm = XArmAPI('192.168.1.201')  # Change for robot's IP
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

# Initial position
x_pos = 200
arm.set_position(246.6, -30.6, 236, 176.6, -2.2, -1.5, speed=50, wait=True)

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Determine the dominant emotion
        emotion = result[0]['dominant_emotion']

        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


        if emotion == 'happy':
          arm.tcp_acc = 2000
          arm.tcp_speed = 300
          arm.angle_acc = 2000
          arm.angle_speed = 300
          arm.set_servo_angle(angle=[-7.0, -35.1, -7.3, -4.9, 40.1, -1.7], speed=arm.angle_speed, mvacc=arm.angle_acc, wait=False, radius=0.0)
          arm.set_position(*[246.6, -30.6, 198.9, 176.6, -2.2, -1.5], speed=arm.tcp_speed, mvacc=arm.tcp_acc, radius=-1.0, wait=True)
          arm.move_circle([271.6, -5.6, 198.9, 176.6, -2.2, -1.5], [271.6, -55.6, 198.9, 176.6, -2.2, -1.5], float(180) / 360 * 100, speed=arm.tcp_speed, mvacc=arm.tcp_acc, wait=True)
          arm.set_servo_angle(angle=[-7.0, -35.1, -7.3, -4.9, 40.1, -1.7], speed=arm.angle_speed, mvacc=arm.angle_acc, wait=False, radius=0.0)
          #arm.set_cgpio_digital(0, 1, delay_sec=0)

        if emotion == 'sad':
          arm.tcp_acc = 2000
          arm.tcp_speed = 300
          arm.angle_acc = 2000
          arm.angle_speed = 300
          arm.set_servo_angle(angle=[-7.0, -35.1, -7.3, -4.9, 40.1, -1.7], speed=arm.angle_speed, mvacc=arm.angle_acc, wait=False, radius=0.0)
          arm.set_position(*[246.6, -30.6, 198.9, 176.6, -2.2, -1.5], speed=arm.tcp_speed, mvacc=arm.tcp_acc, radius=-1.0, wait=True)
          arm.move_circle([271.6, -55.6, 198.9, 176.6, -2.2, -1.5], [271.6, -5.6, 198.9, 176.6, -2.2, -1.5], float(180) / 360 * 100, speed=arm.tcp_speed, mvacc=arm.tcp_acc, wait=True)
          arm.set_servo_angle(angle=[-7.0, -35.1, -7.3, -4.9, 40.1, -1.7], speed=arm.angle_speed, mvacc=arm.angle_acc, wait=False, radius=0.0)

        if emotion == 'neutral':
          arm.tcp_speed = 300
          arm.tcp_acc = 2000
          arm.angle_acc = 2000
          arm.angle_speed = 300
          arm.set_servo_angle(angle=[-7.0, -35.1, -7.3, -4.9, 40.1, -1.7], speed=arm.angle_speed, mvacc=arm.angle_acc, wait=False, radius=0.0)
          arm.set_position(*[246.6, -30.6, 198.9, 176.6, -2.2, -1.5], speed=arm.tcp_speed, mvacc=arm.tcp_acc, radius=-1.0, wait=True)
          arm.set_position(*[306.6, -30.6, 198.9, 176.6, -2.2, -1.5], speed=arm.tcp_speed, mvacc=arm.tcp_acc, radius=-1.0, wait=True)
          arm.set_servo_angle(angle=[-7.0, -35.1, -7.3, -4.9, 40.1, -1.7], speed=arm.angle_speed, mvacc=arm.angle_acc, wait=False, radius=0.0)

        if emotion == 'surprise':
            arm.tcp_acc = 2000
            arm.tcp_speed = 300
            arm.angle_acc = 2000
            arm.angle_speed = 300
            arm.set_servo_angle(angle=[-7.0, -35.1, -7.3, -4.9, 40.1, -1.7], speed=arm.angle_speed, mvacc=arm.angle_acc, wait=False, radius=0.0)
            arm.set_position(*[246.6, -30.6, 198.9, 176.6, -2.2, -1.5], speed=arm.tcp_speed, mvacc=arm.tcp_acc, radius=-1.0, wait=True)
            arm.move_circle([271.6, -55.6, 198.9, 176.6, -2.2, -1.5], [271.6, -5.6, 198.9, 176.6, -2.2, -1.5], float(360) / 360 * 100, speed=arm.tcp_speed, mvacc=arm.tcp_acc, wait=True)
            arm.set_servo_angle(angle=[-7.0, -35.1, -7.3, -4.9, 40.1, -1.7], speed=arm.angle_speed, mvacc=arm.angle_acc, wait=False, radius=0.0)

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
