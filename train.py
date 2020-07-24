import cv2
import numpy as np
import random
import os
from Model import Unet

def data_gen(img_folder, mask_folder, batch_size):
  c = 0
  n = os.listdir(img_folder) #List of training images
  m = os.listdir(mask_folder)
  g=list(zip(n,m))
  random.shuffle(g)
  n,m=zip(*g)
  
  while (True):
    img = np.zeros((batch_size, l, h, 3)).astype('float')
    mask = np.zeros((batch_size, l, h, 1)).astype('float')

    for i in range(c, c+batch_size): 

      train_img = cv2.imread(img_folder+'/'+n[i])/255
      train_img =  cv2.resize(train_img, (h, l))
      
      img[i-c] = train_img                                                   

      train_mask = cv2.imread(mask_folder+'/'+m[i], cv2.IMREAD_GRAYSCALE)
      train_mask = cv2.resize(train_mask, (h, l))
      train_mask = train_mask.reshape(l, h, 1) 

      mask[i-c] = train_mask

    c+=batch_size
    if(c+batch_size>=len(os.listdir(img_folder))):
      c=0
      g=list(zip(n,m))
      random.shuffle(g)
      n,m=zip(*g)
                 
    yield img, mask
    
    
    
train_frame_path = #path to train frames
train_mask_path = #path to train masks

val_frame_path = #path to val frames
val_mask_path = #path to val masks

# Train the model
train_gen = data_gen(train_frame_path,train_mask_path, batch_size = 4)
val_gen = data_gen(val_frame_path,val_mask_path, batch_size = 4)

NO_OF_EPOCHS = 100

BATCH_SIZE = 4

model  = Unet((l, h))

results = model.fit_generator(train_gen, epochs=NO_OF_EPOCHS, 
                                steps_per_epoch = 44,
                                validation_data=val_gen, 
                                validation_steps=22)


model.save_weights("unet.h5")