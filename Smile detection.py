#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install opencv-python')


# In[2]:


#import opencv
import cv2


# In[3]:


# leverage Haar-casdcade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")


# In[4]:


#create a funtion that will dectect the face
def detect(gray, frame):
    face = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w,h) in face:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,100,200), 3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # eyes detection
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)
        for (ex, ey, ew,eh) in eyes:
            cv2.rectangle(frame, (ex+ey), (ex+ew, ey+eh), (100,255,200), 2)
            
        smile = smile_cascade.detectMultiScale(gray, 1.7, 22)
        for (sx, sy, sw,sh) in smile:
            cv2.rectangle(frame, (sx,sy), (sx+sw, sy+sh), (100,255,200), 2)
    
    return frame


# In[5]:


capture = cv2.VideoCapture(0)

while True:
    _, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    cv2.imshow("Video", canvas)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break
capture.release()
cv2.destroyAllWindows()


# In[ ]:




