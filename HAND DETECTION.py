#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install mediapipe')


# In[2]:


import cv2
import mediapipe as mp


# In[3]:


# identify web cam
cap = cv2.VideoCapture(0)


# In[4]:


#leverage the mediapipe library used for hand detection
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


# In[ ]:


#switch on webcam
while True:
    _, img = cap.read()
    
    #convert image from Bg to RGb
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #apply mediapipe
    results = hands.process(imgRGB)
   
    #print(results.multi_hand_landmarks)
     
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id, lm in enumerate(handlms.landmark):
                #print(id, lm)
                mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)            
                
    cv2.imshow("10Alytics Hand detection project", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# Release thre capture once all the processing is done        
cap.release()
cv2.destroyAllwindows()


# In[ ]:





# In[ ]:




