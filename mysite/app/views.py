from django.shortcuts import render
# from app.camera import camerafeed
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from joblib import load

#loadingModel
model = load('C:/Users/Shree/ProjectBE2023/savedModels/model2.joblib')
# Create your views here.
def home(request):
    return render(request,'home.html')


mpHolistics = mp.solutions.holistic # mediapipe holistics model
mpDrawing = mp.solutions.drawing_utils # Drawing utilitie

def mediapipeDetection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # as openCV take BGR therefore we convert it to RGB
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # we reverse it back from RGB to BGR
    return image, results

def drawLandmark(image, results):
    mpDrawing.draw_landmarks(image, results.face_landmarks, mpHolistics.FACEMESH_CONTOURS) # Draw face connections 
    mpDrawing.draw_landmarks(image, results.pose_landmarks, mpHolistics.POSE_CONNECTIONS) # Draw pose connections 
    mpDrawing.draw_landmarks(image, results.left_hand_landmarks, mpHolistics.HAND_CONNECTIONS) # Draw left hand connections 
    mpDrawing.draw_landmarks(image, results.right_hand_landmarks, mpHolistics.HAND_CONNECTIONS) # Draw right hand connections  right hand connectio

def drawStyleLandmark(image, results):
    mpDrawing.draw_landmarks(image, results.face_landmarks, mpHolistics.FACEMESH_CONTOURS,
                            mpDrawing.DrawingSpec(color = (80,110,10), thickness=1, circle_radius=1),
                            mpDrawing.DrawingSpec(color = (80,110,10), thickness=1, circle_radius=1)
                            ) # Draw face connections 
    mpDrawing.draw_landmarks(image, results.pose_landmarks, mpHolistics.POSE_CONNECTIONS,
                            mpDrawing.DrawingSpec(color = (80,22,10), thickness=2, circle_radius=4),
                            mpDrawing.DrawingSpec(color = (80,44,121), thickness=2, circle_radius=2)
                            ) # Draw pose connections 
    mpDrawing.draw_landmarks(image, results.left_hand_landmarks, mpHolistics.HAND_CONNECTIONS,
                            mpDrawing.DrawingSpec(color = (121,22,76), thickness=2, circle_radius=4),
                            mpDrawing.DrawingSpec(color = (121,44,250), thickness=2, circle_radius=2)
                            ) # Draw left hand connections 
    mpDrawing.draw_landmarks(image, results.right_hand_landmarks, mpHolistics.HAND_CONNECTIONS,
                            mpDrawing.DrawingSpec(color = (121,22,76), thickness=2, circle_radius=4),
                            mpDrawing.DrawingSpec(color = (121,44,250), thickness=2, circle_radius=2)
                            ) # Draw right hand connections 

def extractKeypointsArray(results):
    # extracting keypoints from face, pose, lefthand and righthand and storing them in form of numpy arrays
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    leftHand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rightHand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, leftHand, rightHand])


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('C:/Users/Shree/ProjectBE2023/signLanguageDataset') 

# Actions that we try to detect
actions = np.array(['Afternoon','All','Hello','Hi','Language','Myself','our','present','Sign','thank you','we','you'])

labelMap = {label:num for num, label in enumerate(actions)}
        
def webcam(request):
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.6
    cap = cv2.VideoCapture(0)
    
    with mpHolistics.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()

            #make Detection
            image, results = mediapipeDetection(frame, holistic)
            # print(results)

            #draw Landmark
            # drawLandmark(image, results)

            keypoints = extractKeypointsArray(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                sequence = []
                print(actions[np.argmax(res)])

                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] not in sentence:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])
                
                if len(sentence) > 5:
                    sentence = sentence[-5:]

            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    return render(request,'home.html')
    
