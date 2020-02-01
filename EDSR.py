# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 15:52:06 2020

@author: Georgios
"""

#pretrain the EDSR model

from model_architectures import SR, D2
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model

from data_loader import DataLoader
from evaluator import evaluator

import numpy as np

class EDSR():
    def __init__(self, img_shape, SRscale):
        self.img_shape = img_shape
        self.SRscale = SRscale
        self.target_shape = (SRscale*img_shape[0], SRscale*img_shape[1], img_shape[2])
        
        # Configure data loader
        self.data_loader = DataLoader(img_res=(img_shape[0], img_shape[1]), SRscale=SRscale)
        
        #instantiate the models
        self.SR = SR(self.img_shape)
        self.D = D2(self.target_shape)
        
        #compile discriminator
        self.D.compile(loss='mse', loss_weights = [1], optimizer = Adam(0.0002))
        print(self.D.summary())
        
        #create generator graph
        y = Input(shape = self.img_shape)
        
        fake_Y = self.SR(y)
        valid_Y = self.D(fake_Y)
        
        self.D.trainable = False
        self.Generator = Model(inputs = y, outputs = [valid_Y, fake_Y])
        self.Generator.compile(loss = ['mse', 'mse'], loss_weights = [1, 1], optimizer = Adam(0.0002))
        print(self.Generator.summary())
        
    def train(self, epochs, batch_size=10):
        
        #create adversarial ground truths
        out_shape = (batch_size,) + (self.target_shape[0], self.target_shape[1], 1)
        valid_D = np.ones(out_shape)
        fake_D = np.zeros(out_shape)
        
        #define an evaluator object to monitor the progress of the training
        dynamic_evaluator = evaluator(img_res=self.img_shape, SRscale = self.SRscale)
        for epoch in range(epochs):
            for batch, (_, img_y, img_Y) in enumerate(self.data_loader.load_batch(batch_size)):
                #translate domain y to domain Y
                fake_img_Y = self.SR.predict(img_y)
                
                #train Discriminator
                D_loss_real = self.D.train_on_batch(img_Y, valid_D)
                D_loss_fake = self.D.train_on_batch(fake_img_Y, fake_D)
                D_loss = 0.5 * np.add(D_loss_real, D_loss_fake)
                
                #train Generator
                G_loss = self.Generator.train_on_batch([img_y], [valid_D, img_Y])
                
                print("[Epoch %d/%d] [Batch %d/%d]--[D: %.3f] -- [G_adv: %.3f] [G_rec: %.3f]" % (epoch, epochs,
                      batch, self.data_loader.n_batches, D_loss, G_loss[1], G_loss[2]))
                
                if batch % 25 == 0 and batch!=0:
                    """save the model"""
                    model_name="{}_{}.h5".format(epoch, batch)
                    self.SR.save("pretrained models/"+model_name)
                    print("Epoch: {} --- Batch: {} ---- saved".format(epoch, batch))
                    
                    dynamic_evaluator.model = self.SR
                    dynamic_evaluator.epoch = epoch
                    dynamic_evaluator.batch = batch
                    dynamic_evaluator.perceptual_test(5)
                    
                    sample_mean_ssim = dynamic_evaluator.objective_test(batch_size=250)
                    print("Sample mean SSIM: -------------------  %05f   -------------------" % (sample_mean_ssim))
                

Edsr = EDSR(img_shape=(25,25,3), SRscale=2)
Edsr.train(epochs=10, batch_size=10)