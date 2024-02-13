#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import mediapipe as mp


# In[2]:


# step 2 : identify webcam
cap = cv2.VideoCapture(0) 


# In[3]:


mpPose = mp.solutions.pose
pose = mpPose.Pose()
#pose = mpPose.pose(static_image_mode = False, Upper_body_only = False, smooth_landmarks=True, min_detection_confidence=0.5)
#Draw landmarks module
mpDraw = mp.solutions.drawing_utils


# In[ ]:


# switch on webcam

while True:
    _, img = cap.read()
    
    #convert video/image from BGR to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #Apply the mediapipe pose detection module for detection
    results = pose.process(imgRGB)
    print(results.pose_landmarks)
    
    #draw land marks
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    
    cv2.putText(img, "10Alytics Pose Dectection", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2,(255,255,255), 2)  )
    cv2.imshow("10Alytics Pose Dectection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Realease the catpture once all the processing is done.
cap.realease()
cv2.destroyAllWindows()
    


# In[ ]:




