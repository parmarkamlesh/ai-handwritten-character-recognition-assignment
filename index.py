#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
import matplotlib.pyplot as plt
#time package to calculate time required to train model
import time
#get_ipython().run_line_magic('matplotlib', 'inline')
#import warnings

#warnings.filterwarnings('ignore')


# In[3]:


#get_ipython().system('pwd')


# In[4]:


#MNIST dataset is downloaded in locally 
df = pd.read_csv('Train_UQcUa52/train.csv')
df.head()


# In[5]:


imagepath='Train_UQcUa52/Images/train/'


# In[6]:


#Read/load MNIST data set and scale 
X = np.array([img_to_array(load_img(imagepath+df['filename'][i], target_size=(28,28,1), grayscale=True))
              for i in tqdm(range(df.shape[0]))
              ]).astype('float32')


# In[7]:


y = df['label']


# In[8]:


print(X.shape, y.shape)


# In[9]:


image_index = 0
print(y[image_index])
plt.imshow(X[image_index].reshape(28,28), cmap='Greys')


# In[10]:


image_index = 10
print(y[image_index])
plt.imshow(X[image_index].reshape(28,28), cmap='Greys')


# In[11]:


image_index = 192
print(y[image_index])
plt.imshow(X[image_index].reshape(28,28), cmap='Greys')


# In[12]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=np.array(y))


# In[13]:


print(x_train[0])
x_train /= 255
print(x_train[0])
x_test /= 255


# In[15]:



input_shape = (28,28,1)
output_class = 10


# In[16]:


from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

# define the model
model = Sequential()
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.3))
model.add(Dense(output_class, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')


# In[17]:


# train the model
start_time = time.time()
model.fit(x=x_train, y=y_train, batch_size=32, epochs=30, validation_data=(x_test, y_test))
print("Training model took %s Seconds" % (time.time() - start_time))


# In[18]:


image_index = 10
# print("Original output:",y_test[image_index])
plt.imshow(x_test[image_index].reshape(28,28), cmap='Greys')
pred = model.predict(x_test[image_index].reshape(1,28,28,1))
print("Predicted output:", pred.argmax())


# In[19]:


image_index = 100
# print("Original output:",y_test[image_index])
plt.imshow(x_test[image_index].reshape(28,28), cmap='Greys')
pred = model.predict(x_test[image_index].reshape(1,28,28,1))
print("Predicted output:", pred.argmax())


# In[20]:


plt.figure()
plt.imshow(load_img("four.png", target_size=(28,28,1)))
plt.xlabel('Loaded image')
temp = np.array([img_to_array(load_img("four.png", target_size=(28,28,1), grayscale=True))]).astype('float32')
plt.figure()
plt.imshow(temp[0].reshape(28,28), cmap='Greys')
plt.xlabel('Processed image')
pred = model.predict(temp[0].reshape(1,28,28,1))
print("Predicted index array:", pred)
print("Predicted output:", pred.argmax())


# In[ ]:




