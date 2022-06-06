import os
import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import  Activation, Dropout,SpatialDropout1D, Conv3D, ConvLSTM2D, Conv1D, TimeDistributed, Dense, Input, Conv2D, Flatten, MaxPooling2D, Reshape, Conv2DTranspose, BatchNormalization
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ModelCheckpoint
from PIL import Image, ImageSequence
import pylab as plt
import random
# from google.colab import drive
# drive.mount('/content/gdrive')
#PATH_OF_DATA= '/content/gdrive/"My Drive"/Autoencoders_model_Myrto/autoencoders'
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_absolute_error
import sys
from PIL import GifImagePlugin
from skimage.color import rgb2gray
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

tf.random.set_seed(1234)

window=18

# C O M P L E T E   D A T A S E T
frames=[]
for i in range(2003,2021):
  # img = Image.open('/content/gdrive/My Drive/monthly_day/d_%i.gif' %i)
  # img = Image.open('soil_moisture_new\\USA_C_28x28_5km\\%i.gif' %i)
  # img = Image.open('soil_moisture_new\\USA_CAL_28x28_5km\\%i.gif' %i)
  # img = Image.open('soil_moisture_new\\USA_Idaho_28x28\\%i.gif' %i)
  # img = Image.open('soil_moisture_new\\USA_indiana_28x28\\%i.gif' %i)
  img = Image.open('soil_moisture_new\\USA_Oklahoma_28x28\\%i.gif' %i)
  
  
  
  for frame in ImageSequence.Iterator(img):
    frame_2 = frame.convert('RGB')    #(28, 28, 3)
    frame_2 = np.array(frame_2)       #(28, 28, 3)
    frame_2 = np.sum(frame_2, axis=2) #(28, 28)
    frame_2 = frame_2.reshape(frame_2.shape[0], frame_2.shape[1], 1) #( 28, 28, 1)
    frames.append(frame_2) 
frames = np.array(frames)  #(210, 28, 28, 1)

# T R A I N   D A T A 
train_frames=frames[0:209,:,:,:] #(209, 28, 28, 1)
maxi=np.max(train_frames)
mini=np.min(train_frames)
train_frames = (train_frames-mini)/(maxi-mini) #(209, 28, 28, 1)
X_train, y_train = [], []
for i in range(len(train_frames)-window):
  X_train.append(train_frames[i:i+window,::,::,::])
  y_train.append(train_frames[i+1:i+window+1,::,::,::])
X_train=np.array(X_train)  
y_train=np.array(y_train) 

# T E S T   D A T A
test_frames = frames[len(train_frames)-window:,:,:,:]
test_frames = (test_frames-mini)/(maxi-mini)

# V A L I D A T I O N   D A T A 
r=random.sample(range(len(train_frames)-window-1), 20)
X_val=X_train[r, ::, ::, ::, ::] 
y_val=y_train[r, ::, ::, ::, ::]  
X_train = np.delete(X_train, (r), axis=0) 
y_train = np.delete(y_train, (r), axis=0) 

print(np.shape(X_train))
print(np.shape(y_train))



# E - T C N M O D E L

#E N C O D E R   
inp = Input((window, 28, 28, 1))
e =  TimeDistributed(  Conv2D(32, (4, 4), activation='relu')  )(inp)
e =  TimeDistributed(  MaxPooling2D((2, 2)) )(e)
e =  TimeDistributed(  Conv2D(64, (4, 4), activation='relu')  )(e)
e =  TimeDistributed(  MaxPooling2D((2, 2)) )(e)
e =  TimeDistributed(  Conv2D(64, (4, 4), activation='relu')  )(e)
l_o= TimeDistributed(  Flatten() )(e) 

# T C N -> -> ->  F R O M -> -> -> https://github.com/locuslab/TCN/blob/master/TCN/tcn.py      

num_inputs= 64
num_channels=[64, 49, 49] 
kernel_size=4
dropout=0.3
num_levels=len(num_channels) 
for i in range(num_levels):
    in_channels = num_inputs if i == 0 else num_channels[i-1]
    l = tfa.layers.WeightNormalization(   Conv1D(filters=num_channels[i], kernel_size=kernel_size, padding='causal', dilation_rate =  2 ** i)   )(l_o)
    l = Activation('relu')(l)
    l = SpatialDropout1D(dropout)(l)  
    l = tfa.layers.WeightNormalization(   Conv1D(filters=num_channels[i], kernel_size=kernel_size, padding='causal', dilation_rate =  2 ** i)   )(l)
    l = Activation('relu')(l)
    l = SpatialDropout1D(dropout)(l) 
    if in_channels!=num_channels[i]:
        l_o=Conv1D(filters=num_channels[i], kernel_size=1, padding="same")(l_o)
    l_o=tensorflow.keras.layers.add([l_o, l])
    l_o=Activation('relu')(l_o)
          
#D E C O D E R
d = TimeDistributed(  Reshape((7,7,1)) )(l_o)
d = TimeDistributed(  Conv2DTranspose(64, (4, 4), strides=2, activation='relu', padding='same')  )(d)
d = TimeDistributed(  BatchNormalization()  )(d)
d = TimeDistributed(  Conv2DTranspose(64, (4, 4), strides=2, activation='relu', padding='same')  )(d)
d = TimeDistributed(  BatchNormalization()  )(d)
d = TimeDistributed(  Conv2DTranspose(32, (4, 4), activation='relu', padding='same')  )(d)
decoded = TimeDistributed(  Conv2D(1, (4, 4), activation='sigmoid', padding='same' )  )(d)
ae_ETCN = Model(inp, decoded)
ae_ETCN.summary()



# ConvLSTM model 
inp = Input((window, 28, 28, 1))
e   =  ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True) (inp)
e   =  ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True )(e)
e   =  Conv3D(filters=1, kernel_size=(3, 3, 3), activation='sigmoid', padding='same') (e)
ae_ConvLSTM  =  Model(inp, e)
ae_ConvLSTM.summary()


# A Y T O S A V E   B E S T  M O D E L and F I T  M E T H O D
# best_model_file = "vgg.h5"
# best_model = ModelCheckpoint(best_model_file, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
# ae.compile(optimizer='adam', loss="mse")
# history=ae.fit(X_train, y_train, epochs=2000, callbacks=[best_model], validation_data=(X_val, y_val))
# ae.load_weights(best_model_file)

best_model_file1 = "E-TCN_okl.h5"
# best_model_file1 = "E-TCN_cal.h5"
# best_model_file1 = "E-TCN_idaho.h5"
best_model1 = ModelCheckpoint(best_model_file1, monitor='val_loss', mode='min', verbose=1, save_best_only=True)


best_model_file2 = "ConvLSTM_okl.h5"
# best_model_file2 = "ConvLSTM_cal.h5"
# best_model_file2 = "ConvLSTM_idaho.h5"
best_model2 = ModelCheckpoint(best_model_file2, monitor='val_loss', mode='min', verbose=1, save_best_only=True)


optzr =  Adam(lr=1e-4, beta_1=0.9, beta_2=0.999)


ae_ConvLSTM.compile(loss= "mse",optimizer=optzr)
# history=ae_ConvLSTM.fit(X_train, y_train, epochs=100, batch_size=4,callbacks=[best_model2],validation_data=(X_val, y_val))
ae_ConvLSTM.load_weights(best_model_file2)



ae_ETCN.compile(loss= "mse",optimizer=optzr)
# history=ae_ETCN.fit(X_train, y_train, epochs=100, batch_size=4,callbacks=[best_model1],validation_data=(X_val, y_val))
ae_ETCN.load_weights(best_model_file1)



# best_model_file = "ConvLSTM_cal.h5"
# ae_ConvLSTM.load_weights(best_model_file)


# L O S S E S
# fig = plt.figure()
# my_dpi=96
# plt.plot(history.history['loss'], label="Train Loss")
# plt.plot(history.history['val_loss'], label="Validation Loss")
# plt.xlabel("Epochs",fontsize=12,labelpad=5)
# plt.ylabel("Loss", fontsize=12, labelpad=5)
# plt.legend(fontsize=12)
# plt.show()
# # fig.savefig('Losses.png',dpi=my_dpi*8)

# train_loss_history = history.history['loss']
# train_loss_history = np.asarray( train_loss_history)
# f1 = open("train_loss","w")
# for i in range(train_loss_history.shape[0]):
#     f1.write(str( train_loss_history[i]) + "\n" )
# f1.close()

# val_loss_history = history.history['val_loss']
# val_loss_history = np.asarray(val_loss_history)
# f2 = open("validation_loss","w")
# for i in range(val_loss_history.shape[0]):
#     f2.write(str(val_loss_history[i]) + "\n" )
# f2.close()


# P R E D I C T I O N
new_pos_ETCN=ae_ETCN.predict(test_frames[np.newaxis, 0: window, ::, ::, ::]) #(1, window, 28, 28, num_channels to predict)
new_pos_ConvLSTM=ae_ConvLSTM.predict(test_frames[np.newaxis, 0: window, ::, ::, ::]) #(1, window, 28, 28, num_channels to predict)
# new_pos_ConvLSTM=ae_ETCN.predict(test_frames[np.newaxis, 0: window, ::, ::, ::]) #(1, window, 28, 28, num_channels to predict)

# P L O T S
# fig = plt.figure(figsize=(10, 5))
# ax = fig.add_subplot(121)
# plt.imshow((new_pos[0,-1,::,::,0])) 
# ax = fig.add_subplot(122)
# plt.imshow(test_frames[window,::,::, 0])
# plt.show()
# plt.savefig('prediction1.png' )

test=test_frames[window,::,::, 0].flatten()
# test=np.clip(test,0.4,0.75)
pred_ETCN=new_pos_ETCN[0,-1,::,::,0].flatten()

print('E-TCN')
print(pearsonr(test, pred_ETCN))
print(mean_absolute_error(test, pred_ETCN))
print(mean_squared_error(test, pred_ETCN))

pred_convlstm=new_pos_ConvLSTM[0,-1,::,::,0].flatten()

print('ConvLSTM')
print(pearsonr(test, pred_convlstm))
print(mean_absolute_error(test, pred_convlstm))
print(mean_squared_error(test, pred_convlstm))



plt.scatter(pred_ETCN,test,s=3.0,label='Embedded TCN', alpha=0.5)
plt.scatter(pred_convlstm,test,s=3.0,label='ConvLSTM', alpha=0.5)
plt.scatter(np.arange(0.5,0.8,0.01),np.arange(0.5,0.8,0.01),s=1.0,color = 'blue')
plt.legend()
plt.xlabel('Predicted values of soil moisture')
plt.ylabel('True values of soil moisture')
plt.show()
