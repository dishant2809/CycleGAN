import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import Sequential
from keras import Model
import os
import cv2
import numpy as np
import tqdm
from cyclegan import *

    
def train(mgen ,pgen, mdisc, pdisc , mse, l1, disc_opt, gen_opt, data):
    m,p = data
    with tf.GradientTape() as tape:
        fake_m = mgen(p, training=True)
        cycle_p = pgen(fake_m, training=True)
            
        fake_p = pgen(m, training=True)
        cycle_m = mgen(fake_p, training=True)
            
        same_m = mgen(m, training=True)
        same_p = pgen(p, training=True)
            
        disc_real_m = mdisc(m, training=True)
        disc_real_p = pdisc(p, training=True)
            
        disc_fake_m = mdisc(fake_m, training=True)
        disc_fake_p = pdisc(fake_p, training=True)
            
        mgen_loss = gen_loss(disc_fake_m)
        pgen_loss = gen_loss(disc_fake_p)
            
        total_cycle_loss = cycle_loss(m, cycle_m, lambda_cycle) + cycle_loss(p, cycle_p, lambda_cycle)
            
        total_mgen_loss = mgen_loss + total_cycle_loss +identity_loss_fn(m, same_m, lambda_cycle)
        total_pgen_loss = pgen_loss + total_cycle_loss +identity_loss_fn(p, same_p, lambda_cycle)
            
        mdisc_loss = disc_loss(disc_real_m, disc_fake_m)
        pdisc_loss = disc_loss(disc_real_p, disc_fake_p)
            
    mgen_gradient = tape.gradient(total_mgen_loss, mgen.trainable_variables)
    pgen_gradient = tape.gradient(total_pgen_loss, pgen.trainable_variables)
        
    mdisc_gradient = tape.gradient(mdisc_loss, mdisc.trainable_variables)
    pdisc_gradient = tape.gradient(pdisc_loss, pdisc.trainable_variables)
        
    mgen_optimizer.apply_gradient(zip(mgen_gradient, mgen.trainable_variables))
    pgen_optimizer.apply_gradient(zip(pgen_gradient, pgen.trainable_variables))
        
    mdisc_optimizer.apply_gradient(zip(mdisc_gradient, mdisc.trainable_variables))
    pdisc_optimizer.apply_gradient(zip(pdisc_gradient, pdisc.trainable_variables))
        
    return {
        'mgen_loss':mgen_loss,
        'pgen_loss':pgen_loss,
        'mdisc_loss':mdisc_loss,
        'pdisc_loss':pdisc_loss
        }



mgen = Generator()
pgen = Generator()
mdisc = Discriminator()
pdisc = Discriminator()
disc_opt = tf.keras.optimizers.Adam()
gen_opt = tf.keras.optimizers.Adam()


def identity_loss(real_image, same_image, LAMBDA):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss

    
def calc_cycle_loss(real_image, cycled_image, LAMBDA):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return LAMBDA * loss1




for epoch in range(2):
    train(mgen, pgen, mdisc, pdisc, identity_loss, calc_cycle_loss, disc_opt,gen_opt , (h1,z1))

























'''def train(mgen ,pgen, mdisc, pdisc , mse, l1, disc_opt, gen_opt, data):
    loop = tqdm(data, leave=True)
    
    for idx, (m,p) in enumerate(loop):
        with tf.GradientTape() as g_tape:
            fake_m = mgen(p)
            d_m_real = mdisc(m)
            d_m_fake = mdisc(fake_m, training=True)
            m_real = tf.math.reduce_all(d_m_real)
            m_fake = tf.math.reduce_all(d_m_fake)
            d_m_real_loss = mse(d_m_real, np.ones(256))
            d_m_fake_loss = mse(d_m_fake, np.zeros(256))
            d_m_loss = d_m_real_loss + d_m_fake_loss
            
            fake_p = pgen(m)
            d_p_real = pdisc(m)
            d_p_fake = pdisc(fake_p, training=True)
            p_real = tf.math.reduce_all(d_p_real)
            p_fake = tf.math.reduce_all(d_p_fake)
            d_p_real_loss = mse(d_p_real, np.ones(256))
            d_p_fake_loss = mse(d_p_fake, np.zeros(256))
            d_p_loss = d_p_real_loss + d_p_fake_loss
            
            
            d_loss = (d_p_loss + d_m_loss)/2
            
            
            
            
            
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = disc_H(fake_horse)
            D_Z_fake = disc_Z(fake_zebra)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            # cycle loss
            cycle_zebra = gen_Z(fake_horse)
            cycle_horse = gen_H(fake_zebra)
            cycle_zebra_loss = l1(zebra, cycle_zebra)
            cycle_horse_loss = l1(horse, cycle_horse)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_zebra = gen_Z(zebra)
            identity_horse = gen_H(horse)
            identity_zebra_loss = l1(zebra, identity_zebra)
            identity_horse_loss = l1(horse, identity_horse)


         G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_zebra_loss * config.LAMBDA_CYCLE
                + cycle_horse_loss * config.LAMBDA_CYCLE
                + identity_horse_loss * config.LAMBDA_IDENTITY
                + identity_zebra_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_horse*0.5+0.5, f"saved_images/horse_{idx}.png")
            save_image(fake_zebra*0.5+0.5, f"saved_images/zebra_{idx}.png")

        loop.set_postfix(H_real=H_reals/(idx+1), H_fake=H_fakes/(idx+1))
        '''