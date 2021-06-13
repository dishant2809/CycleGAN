import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential
from keras import Model
import os
import cv2
import numpy as np


def downsample(filters, size, strides ,batchnorm=True):
    
    x = Sequential()
    
    x.add(Conv2D(filters, size, strides=strides ,padding='same',use_bias=False))
    
    if batchnorm:
        x.add(BatchNormalization())
        
    x.add(LeakyReLU())
    
    return x


def upsample(filters, size, strides ,dropout=False):
    
    x = Sequential()
    
    x.add(Conv2DTranspose(filters, size,strides =strides ,padding='same',use_bias=False))
    
    if dropout==True:
        x.add(Dropout(0.2))
        
    x.add(ReLU())
    
    return x


def Generator():
    
    i = Input(shape=(256,256,3))
    
    x = downsample(32, 5, 2)(i)
    
    x = downsample(64, 5, 2)(x)
    
    x = downsample(64,5,2)(x)
    
    x = downsample(128,5,2)(x)
    
    x = downsample(256,5,2)(x)
    
    x = downsample(512,5,2)(x)
    
    x = downsample(512,5,2)(x)
    
    x = downsample(512,5,2)(x)
    
    x = upsample(512, 5, 2)(x)
    
    x = upsample(256, 5, 2)(x)
    
    x = upsample(256, 5, 2)(x)
    
    x = upsample(256, 5, 2)(x)
    
    x = upsample(128, 5, 2)(x)
    
    x = upsample(64, 5, 2)(x)
    
    x = upsample(32, 5, 2)(x)
    
    x = upsample(16, 5, 2)(x)
    
    x = Conv2D(3, 5, strides=1, padding='same')(x)
    
    model = Model(i, x)
    
    return model


def Discriminator():
    
    i = Input(shape=(256,256,3))
    
    x = downsample(32, 5, 2)(i)
    
    x = downsample(64, 5, 2)(x)
    
    x = downsample(128, 5, 2)(x)
    
    x = downsample(256, 5, 2)(x)
    
    x = downsample(128, 5, 2)(x)
    x = downsample(64, 5, 2)(x)
    x = downsample(32, 5, 2)(x)
    x = downsample(1, 5, 2)(x)
    
    model = Model(i,x)
    
    return model

    
'''
class CycleGAN(tf.Module):
    def __init__(self,mgen, pgen, mdisc, pdisc):
        super(CycleGAN, self).__init__()
        self.mgen = mgen
        self.pgen = pgen
        self.mdisc = mdisc
        self.pdisc = pdisc
        self.lambda_cycle = 10
        
    def compile(self,gen_loss,disc_loss,mdisc_optimizer,mgen_optimizer,pdisc_optimizer,pgen_optimizer,identity_loss,cycle_loss):
        super(CycleGAN).compile(self)
        self.gen_loss = gen_loss
        self.disc_loss = disc_loss
        self.mdisc_optimizer = mdisc_optimizer
        self.mgen_optimizer = mgen_optimizer
        self.pdisc_optimizer = pdisc_optimizer
        self.pgen_optimizer = pgen_optimizer
        self.identity_loss = identity_loss
        self.cycle_loss = cycle_loss

        
    def train_step(self, data):
        m,p = data
        
        with tf.GradientTape() as tape:
            fake_m = self.mgen(p, training=True)
            cycle_p = self.pgen(fake_m, training=True)
            
            fake_p = self.pgen(m, training=True)
            cycle_m = self.mgen(fake_p, training=True)
            
            same_m = self.mgen(m, training=True)
            same_p = self.pgen(p, training=True)
            
            disc_real_m = self.mdisc(m, training=True)
            disc_real_p = self.pdisc(p, training=True)
            
            disc_fake_m = self.mdisc(fake_m, training=True)
            disc_fake_p = self.pdisc(fake_p, training=True)
            
            mgen_loss = self.gen_loss(disc_fake_m)
            pgen_loss = self.gen_loss(disc_fake_p)
            
            total_cycle_loss = self.cycle_loss(m, cycle_m, self.lambda_cycle) + self.cycle_loss(p, cycle_p, self.lambda_cycle)
            
            total_mgen_loss = mgen_loss + total_cycle_loss +self.identity_loss_fn(m, same_m, self.lambda_cycle)
            total_pgen_loss = pgen_loss + total_cycle_loss +self.identity_loss_fn(p, same_p, self.lambda_cycle)
            
            mdisc_loss = self.disc_loss(disc_real_m, disc_fake_m)
            pdisc_loss = self.disc_loss(disc_real_p, disc_fake_p)
            
        mgen_gradient = tape.gradient(total_mgen_loss, self.mgen.trainable_variables)
        pgen_gradient = tape.gradient(total_pgen_loss, self.pgen.trainable_variables)
        
        mdisc_gradient = tape.gradient(mdisc_loss, self.mdisc.trainable_variables)
        pdisc_gradient = tape.gradient(pdisc_loss, self.pdisc.trainable_variables)
        
        self.mgen_optimizer.apply_gradient(zip(mgen_gradient, self.mgen.trainable_variables))
        self.pgen_optimizer.apply_gradient(zip(pgen_gradient, self.pgen.trainable_variables))
        
        self.mdisc_optimizer.apply_gradient(zip(mdisc_gradient, self.mdisc.trainable_variables))
        self.pdisc_optimizer.apply_gradient(zip(pdisc_gradient, self.pdisc.trainable_variables))
        
        return {
            'mgen_loss':mgen_loss,
            'pgen_loss':pgen_loss,
            'mdisc_loss':mdisc_loss,
            'pdisc_loss':pdisc_loss
            }
    
'''

t1 = os.listdir('C:/Users/vedant/ML&AI/Dishant/CycleGAN/trainA')
t2 = os.listdir('C:/Users/vedant/ML&AI/Dishant/CycleGAN/trainB')

h1=[]
z1=[]
for i in t1:
    a = cv2.imread('C:/Users/vedant/ML&AI/Dishant/CycleGAN/trainA/'+i)
    a = cv2.resize(a,(256,256))
    h1.append(a)

for i in t2:
    a = cv2.imread('C:/Users/vedant/ML&AI/Dishant/CycleGAN/trainB/'+i)
    a = cv2.resize(a,(256,256))
    z1.append(a)

h1 = np.array(h1)
z1 = np.array(z1)


#model.compile(tf.keras.losses.BinaryCrossentropy(), tf.keras.losses.BinaryCrossentropy(), tf.keras.optimizers.Adam(),  tf.keras.optimizers.Adam(),  tf.keras.optimizers.Adam(),  tf.keras.optimizers.Adam(), identity_loss, calc_cycle_loss)

#model.fit(zip(h1,z1), batch_size=64, epochs=1)
