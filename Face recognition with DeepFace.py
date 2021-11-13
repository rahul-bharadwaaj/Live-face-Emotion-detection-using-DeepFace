#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv


# In[6]:


from deepface import DeepFace


# In[18]:


img=cv.imread(r'C:\Users\RAHUL BHARADWAAJ\OneDrive\Desktop\CV\happy 1.jpg')


# In[3]:


import matplotlib.pyplot as plt


# In[19]:


plt.imshow(img) #BGR


# In[20]:


plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))


# In[21]:


pridictions=DeepFace.analyze(img)


# In[9]:


pridictions


# In[22]:


pridictions['dominant_emotion']


# In[28]:


faceCascade=cv.CascadeClassifier(cv.data.haarcascades+'./haarcascade_frontalface_default.xml')


# In[29]:


gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(gray,1.1,4)
for(x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


# In[41]:


plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))


# In[42]:


font=cv.FONT_HERSHEY_PLAIN
cv.putText(img,
          pridictions['dominant_emotion'],
          (0,50),
          font,1,
          (255,0,0),
          2,
          cv.LINE_4)


# In[43]:


plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))


# In[44]:


img=cv.imread(r"C:\Users\RAHUL BHARADWAAJ\OneDrive\Desktop\CV\anger.png")


# In[45]:


plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))


# In[51]:


pridictions=DeepFace.analyze(img)


# In[52]:


pridictions['dominant_emotion']


# In[61]:


font=cv.FONT_HERSHEY_PLAIN
cv.putText(img,
          pridictions['dominant_emotion'],
          (0,50),
          font,1,
          (255,0,0),
          2,
          cv.LINE_4)


# In[62]:


plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))


# In[63]:


cap=cv.VideoCapture(1)
if not cap.isOpened():
    cap=cv.VideoCapture(0)
if not cap.isOpened():
    raise IOError("cannot open webcam")
while True:
    ret,frame=cap.read()
    result=DeepFace.analyze(frame,actions=['emotion'],enforce_detection=False)
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray,1.1,4)
    for(x,y,w,h) in faces:
         cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
   
    font=cv.FONT_HERSHEY_PLAIN
    cv.putText(img,
          pridictions['dominant_emotion'],
          (0,50),
          font,3,
          (0,0,255),
          2,
          cv.LINE_4)
    cv.imshow('demo_video',frame)
    if cv.waitKey(2) & 0xFF==ord('q'):
        break
cap.release()
cv.destroyAllWindows()


# In[ ]:




