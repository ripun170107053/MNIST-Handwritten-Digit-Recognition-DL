import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from matplotlib.pyplot import imshow
from PIL import Image
import cv2 as cv
import keras
from keras.layers import Conv2D, Input, LeakyReLU, Dense, Activation, Flatten, Dropout, MaxPool2D
from keras import models
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import math,time
import seaborn as sns
#Training and test sets
training_set=pd.read_csv('train.csv/train.csv')
test_set=pd.read_csv('test.csv/test.csv')
#all images are flattened..need to reshaped to 2d

#split training_set into two parts..use 2nd part like a test set with known result..
#the original test set is test set w/o known result

trsize= training_set.shape[0]
valsize= int(trsize*0.1)
ts=training_set.values
ts=ts[:int(trsize*0.9),1:]
train_x=ts
train_x=train_x.reshape(train_x.shape[0],28,28,1)
ts=training_set.values
ts=ts[:int(trsize*0.9),0:1]
train_y=ts.reshape(ts.shape[0],1)
ts=training_set.values
val_x=ts[int(trsize*0.9):,1:]
val_x=val_x.reshape(4200,28,28,1)
ts=training_set.values
ts=ts[int(trsize*0.9):,0:1]
val_y=ts.reshape(ts.shape[0],1)
train_x=train_x/255
val_x=val_x/255
test_set=test_set/255



"""
np.random.seed(1)
training_set=training_set.iloc[np.random.permutation(len(training_set))]
training_set.head()
sample_size=training_set.shape[0]
val_size=int(sample_size*0.1)
train_x=np.asarray(training_set.iloc[:sample_size-val_size,1:]).reshape([sample_size-val_size,28,28,1])
train_y=np.asarray(training_set.iloc[:sample_size-val_size,0]).reshape([sample_size-val_size,1])
val_x=np.asarray(training_set.iloc[sample_size-val_size:,1:]).reshape([val_size,28,28,1])
val_y=np.asarray(training_set.iloc[sample_size-val_size:,0]).reshape([val_size,1])
test_set=np.asarray(test_set.iloc[:,:]).reshape([-1,28,28,1])
train_x=train_x/255
val_x=val_x/255
test_set=test_set/255
"""


#scaling pixel values(0 to 255) to (0,1) for easier learning

""" checking if train set/val_x is balanced. it is
xx=[0,1,2,3,4,5,6,7,8,9]
yy=[0]*10
for i in range(37800):
    z=int(train_y[i][0])
    #z=int(val_y[i][0])
    yy[z]+=1
for i in range(10):
    plt.text(i,yy[i],str(yy[i]))

plt.bar(xx,yy)
plt.show()
"""
"""
#vitualize the digits 
rows=2
cols=2
fig=plt.figure(figsize=(cols,rows))
plt.imshow(train_x[444].reshape([28,28]))

"""
#CNN
# Block 1
model=models.Sequential()
model.add(Conv2D(32,3, padding  ="same",input_shape=(28,28,1)))
model.add(LeakyReLU())
model.add(Conv2D(32,3, padding  ="same"))
model.add(LeakyReLU())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Block 2
model.add(Conv2D(64,3, padding  ="same"))
model.add(LeakyReLU())
model.add(Conv2D(64,3, padding  ="same"))
model.add(LeakyReLU())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation="sigmoid"))
initial_lr = 0.001
loss = "sparse_categorical_crossentropy"
model.compile(Adam(lr=initial_lr), loss=loss ,metrics=['accuracy'])

epochs = 10
batch_size = 256
history_1 = model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs,validation_data=[val_x,val_y])

#SVM

#KernelSVM
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

linear_svm=SVC(kernel='linear')
train_data=training_set
y=train_data['label']
X=train_data.drop(columns='label')
X=X/255.0
test_set=test_set/255
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
X_scaled =scale(X)
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.3,train_size=0.2,random_state=0)
from sklearn import metrics
from sklearn.metrics import confusion_matrix
print(metrics.accuracy_score(test_set,y_pred_lin_svm))



"""
#vitualizing training performance
run1=history_1
f=plt.figure(figsize=(20,7))
f.add_subplot(121)
plt.plot(run1.epoch,run1.history['accuracy'],label='acc')
plt.plot(run2.epoch,run2.history['accuracy'],label='acc')
plt.plot(run1.epoch,run1.history['val_accuracy'],label='valacc')
plt.plot(run2.epoch,run2.history['val_accuracy'],label='valacc')
plt.plot(run1.epoch,run1.history['loss'],label='valacc')
plt.plot(run2.epoch,run2.history['loss'],label='valacc')

"""
ans=classifier.predict(test_x)
a1=ans[:5]
b4=test_x[4].reshape(28,28)

classifier=model
model_json=classifier.to_json()
with open("classifier.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights("classifierW.h5")
print("Saved model to disk")


#set the architechture of the model then use below code to load the predefined weights compiled on the first run
model.load_weights('classifierW.h5')
ans=model.predict(test_X)

#confusion matrix
val_p=np.argmax(model.predict(val_x),axis=1)
#^indices of max value present in pred for each row
cm=np.zeros([10,10])
err=0
for i in range(val_x.shape[0]):
    cm[val_y[i],val_p[i]]+=1
    if val_y[i]!=val_p[i]:
        err+=1


val_p_1=[i for i in range(28000)]
val_p_2=np.argmax(model.predict(test_set),axis=1)
val_r=[val_p_2[i] for i in range(len(val_p_2))]
ress=[]
for i in range(28000):
    ress.append([int(i+1),int(val_r[i])])
np.savetxt('haha.csv',ress,fmt='%0.0f',delimiter=',')

""" to csv
val_p_1=[i for i in range(28000)]
val_p_2=np.argmax(pred,axis=1)
val_r=[val_p_2[i] for i in range(28000)]
ress=[]
for i in range(28000):
    ress.append([int(i+1),int(val_r[i])])
np.savetxt('haha.csv',ress,fmt='%0.0f',delimiter=',')
"""
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
#img augmentation
datagen=ImageDataGenerator(
    rotation_range=10,zoom_range=0.1, width_shift_range=(0.1), height_shift_range=(0.1)
    )
datagen.fit(train_x)
lrr=ReduceLROnPlateau(monitor='val_accuracy',patience=2,factor=0.5)
history_2=model.fit(datagen.flow(train_x, train_y, batch_size=32),
          steps_per_epoch=len(train_x) / 32, epochs=10, validation_data=[val_x,val_y], callbacks=[lrr])
