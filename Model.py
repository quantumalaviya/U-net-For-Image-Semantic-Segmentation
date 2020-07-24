import tensorflow as tf
from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, BatchNormalization, ReLU, Dropout
from keras.optimizers import Adam

def dice_loss(y_true, y_pred):
  numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
  denominator = tf.reduce_sum(y_true + y_pred, axis=-1)

  return 1 - (numerator + 1) / (denominator + 1)


def Unet(Shape):
    input = Input(shape = Shape)
    conv1 = Conv2D(16, (3,3), padding = 'same', bias_initializer = 'he_normal')(input)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)
    conv1 = Conv2D(16, (3,3), padding = 'same', bias_initializer = 'he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)
    
    maxpool1 = MaxPooling2D((2,2))(conv1)
    maxpool1 = Dropout(0.2)(maxpool1) 
    
    
    conv2 = Conv2D(32, (3,3), padding = 'same', bias_initializer = 'he_normal')(maxpool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)
    conv2 = Conv2D(32, (3,3), padding = 'same', bias_initializer = 'he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)

    maxpool2 = MaxPooling2D((2,2))(conv2)
    maxpool2 = Dropout(0.2)(maxpool2)    
    
    conv3 = Conv2D(64, (3,3), padding = 'same', bias_initializer = 'he_normal')(maxpool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)
    conv3 = Conv2D(64, (3,3), padding = 'same', bias_initializer = 'he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = ReLU()(conv3)
    
    maxpool3 = MaxPooling2D((2,2))(conv3)    
    maxpool3 = Dropout(0.2)(maxpool3)
    
    conv4 = Conv2D(128, (3,3), padding = 'same', bias_initializer = 'he_normal')(maxpool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = ReLU()(conv4)
    conv4 = Conv2D(128, (3,3), padding = 'same', bias_initializer = 'he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = ReLU()(conv4)
    
    maxpool4 = MaxPooling2D((2,2))(conv4)
    maxpool4 = Dropout(0.2)(maxpool4)
    
    conv5 = Conv2D(256, (3,3), padding = 'same', bias_initializer = 'he_normal')(maxpool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = ReLU()(conv5)
    conv5 = Conv2D(256, (3,3), padding = 'same', bias_initializer = 'he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = ReLU()(conv5)
    

    up1 = Conv2DTranspose(128, (3,3), strides = (2,2), padding = 'same', bias_initializer = 'he_normal')(conv5)
    up1 = concatenate([up1, conv4])
    up1 = Dropout(0.2)(up1)
    
    up1 = Conv2D(128, (3,3), padding = 'same', bias_initializer = 'he_normal')(up1)
    up1 = BatchNormalization()(up1)
    up1 = ReLU()(up1)
    up1 = Conv2D(128, (3,3), padding = 'same', bias_initializer = 'he_normal')(up1)
    up1 = BatchNormalization()(up1)
    up1 = ReLU()(up1) 

    up2 = Conv2DTranspose(64, (3,3), strides = (2,2), padding = 'same', bias_initializer = 'he_normal')(up1)

    up2 = concatenate([up2, conv3])
    up2 = Dropout(0.2)(up2)
    
    up2 = Conv2D(64, (3,3), padding = 'same', bias_initializer = 'he_normal')(up2)
    up2 = BatchNormalization()(up2)
    up2 = ReLU()(up2)
    up2 = Conv2D(64, (3,3), padding = 'same', bias_initializer = 'he_normal')(up2)
    up2 = BatchNormalization()(up2)
    up2 = ReLU()(up2) 
    
    
    up3 = Conv2DTranspose(32, (3,3), strides = (2,2), padding = 'same', bias_initializer = 'he_normal')(up2)
    up3 = concatenate([up3, conv2])
    up3 = Dropout(0.2)(up3)
    
    up3 = Conv2D(32, (3,3), padding = 'same', bias_initializer = 'he_normal')(up3)
    up3 = BatchNormalization()(up3)
    up3 = ReLU()(up3)
    up3 = Conv2D(32, (3,3), padding = 'same', bias_initializer = 'he_normal')(up3)
    up3 = BatchNormalization()(up3)
    up3 = ReLU()(up3) 
    
    
    up4 = Conv2DTranspose(16, (3,3), strides = (2,2), padding = 'same', bias_initializer = 'he_normal')(up3)
    up4 = concatenate([up4, conv1])
    up4 = Dropout(0.2)(up4)
    
    up4 = Conv2D(16, (3,3), padding = 'same', bias_initializer = 'he_normal')(up4)
    up4 = BatchNormalization()(up4)
    up4 = ReLU()(up4)
    up4 = Conv2D(16, (3,3), padding = 'same', bias_initializer = 'he_normal')(up4)
    up4 = BatchNormalization()(up4)
    up4 = ReLU()(up4)
    
    output = Conv2D(1, (1, 1), activation='sigmoid') (up4)

    model = Model(inputs = [input], outputs = [output])
    
    model.compile(
                Adam(),
                loss = dice_loss,
                metrics = ['accuracy']
            )  

    return model  


