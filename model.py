# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 21:34:57 2020

@author: Georgios
"""

#Implementation of the MCinCGAN paper

#laod required modules
import time
import numpy as np

#load functions and classes from other .py files within the repository
from data_loader import DataLoader
from evaluator import evaluator 
from loss_functions import total_variation
from model_architectures import G1, D1, G2, SR, D2, G3, blur

#keras
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.callbacks import ModelCheckpoint, TensorBoard

#visualisation packages
import matplotlib.pyplot as plt

class CINGAN():
    def __init__(self, img_shape, SRscale=2):
        # Input shape
        self.SRscale=SRscale
        self.img_shape = img_shape
        self.target_res = (self.SRscale*self.img_shape[0], self.SRscale*self.img_shape[1], self.img_shape[2])
        
        # Configure data loader
        self.data_loader = DataLoader(img_res=(img_shape[0], img_shape[1]), SRscale=SRscale)
        
        
        #Training will take place in 2 steps. 
        
        #In the first step a combined which contains all the generators will be updated
        #In the second step all the discriminators will be updated
        
        #For this reason we define a combined model which contains all the generator mapping
        #The discriminators are defined seperately
        
        self.G1 = G1(img_shape)
        self.D1 = D1(img_shape)
        self.G2 = G2(img_shape)
        self.blur = blur(img_shape)
        self.SR = SR(img_shape)
        self.SR.load_weights('pretrained models/128_8.h5')
        self.D2 = D2(self.target_res)
        self.G3 = G3(self.target_res)


        #compile the discriminators
        optimizer = Adam(0.0002)
        self.D1.compile(loss='mse', loss_weights=[1], optimizer=optimizer, metrics=['accuracy'])
        print(self.D1.summary())
        self.D2.compile(loss='mse', loss_weights=[1], optimizer=optimizer, metrics=['accuracy'])
        
        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #    (combined model)
        #-------------------------

        # Input images from both domains
        x = Input(shape = img_shape)
        y = Input(shape = img_shape)
        Y = Input(shape = self.target_res)
        
        #-------------denoising network---------------
        # Translate images to the other domain
        fake_y = self.G1(x) #L_tv(1)
        
        #cycle-consinstent image
        cyc_x = self.G2(fake_y) #L_cyc(1)
        
        #pass fake_y through the blurring kernel
        blur_fake_y = self.blur(fake_y) #L_blur(1)
        
        #pass fake_y through discriminator D1
        valid_y = self.D1(fake_y) #L_GAN(1)
        
        #------------------SR network----------------------
        SR_fake_Y = self.SR(fake_y) #L_tv(2)
        valid_Y = self.D2(SR_fake_Y) #L_GAN(2)
        cyc_x_2 = self.G3(SR_fake_Y) #L_cyc(2)
        
        # Freeze the discriminators
        self.D1.trainable = False
        self.D2.trainable = False
        
        """loss_relative parameters"""
        #Denoising network paramaeters
        w1=10 #relative importance cycle constintency 
        w2=20 #relative importance of conservation of color distribution
        w3=1 #relative importance of total variation
        
        #1st SR network parameters
        l1 = 10 #relative importance cycle consistency in 1st cycle
        l2 = 1 #relative importance of total variation

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[x, y, Y] ,
                              outputs=[valid_y, cyc_x, blur_fake_y, fake_y, 
                                       valid_Y, cyc_x_2, SR_fake_Y])
        
        self.combined.compile(loss=['mse', 'mse', 'mse', total_variation, 
                                    'mse', 'mse', total_variation],
                            loss_weights=[1, w1, w2, w3, 
                                          1, l1, l2],
                            optimizer=optimizer)
        print(self.combined.summary())
        
        #logger settings
        self.training = []
        
        self.D1_loss = []
        self.D2_loss = []
        self.G1_adv = []
        self.SR_adv = []
        
        self.cyc1 = []
        self.blur1 = []
        self.tv1 = []
        
        self.cyc2 = []
        self.tv2 = []
        
        self.ssim_eval_time = []
        self.ssim = []
    
    def log(self,):
        fig, axs = plt.subplots(2, 3)
        
        ax = axs[0,0]
        ax.plot(self.training, self.D1_loss, label="D1 adv loss")
        ax.plot(self.training, self.G1_adv, label="G1 adv loss")
        ax.legend()
        ax.set_title("Adv losses (1st)")
        
        ax = axs[1,0]
        ax.plot(self.training, self.D2_loss, label="D2 adv loss")
        ax.plot(self.training, self.SR_adv, label="SR adv loss")
        ax.legend()
        ax.set_title("Adv losses (2nd)")
        
        ax = axs[0,1]
        ax.plot(self.training, self.cyc1, label = "cyc1")
        ax.plot(self.training, self.cyc2, label = "cyc2")
        ax.legend()
        ax.set_title("cyclic losses")
        
        ax = axs[1,1]
        ax.plot(self.training, self.tv1, label = "TV 1")
        ax.plot(self.training, self.tv2, label = "TV 2")
        ax.set_title("Total Variation losses")
        
        ax = axs[0,2]
        ax.plot(self.training, self.blur1, label = "blur loss 1")
        ax.legend()
        ax.set_title("Blur loss 1")
        fig.savefig("progress/log.png")
        
        fig, axs = plt.subplots(1,1)
        ax=axs
        ax.plot(self.ssim_eval_time, self.ssim)
        ax.set_title("SSIM evolution")
        fig.savefig("progress/ssim_evolution.png")
        
        plt.close("all")
        
    def train(self, epochs, batch_size=10, sample_interval=50):
        #every sample_interval batches, the model is saved and sample images are generated and saved
        
        start_time = time.time()

        """ Adversarial loss ground truths for patchGAN discriminators"""
        
        # Calculate output shape of the patchGAN discriminators based on the target shape
        len_x = self.img_shape[0]
        len_y = self.img_shape[1]
        self.D1_out_shape = (len_x, len_y, 1)
        #define the adversarial ground truths for D1
        valid_D1 = np.ones((batch_size,) + self.D1_out_shape)
        fake_D1 = np.zeros((batch_size,) + self.D1_out_shape)
        print("valid D1: ", valid_D1.shape)
        #similarly for D2
        len_x = self.target_res[0]
        len_y = self.target_res[1]
        self.D2_out_shape = (len_x, len_y, 1)
        
        valid_D2 = np.ones((batch_size,) + self.D2_out_shape)
        fake_D2 = np.zeros((batch_size,) + self.D2_out_shape)
        
        #define an evaluator object to monitor the progress of the training
        dynamic_evaluator = evaluator(img_res=self.img_shape, SRscale = self.SRscale)
        for epoch in range(epochs):
            for batch, (img_x, img_y, img_Y) in enumerate(self.data_loader.load_batch(batch_size)):

                # Update the discriminators 

                # Make the appropriate generator translations for discriminator training
                fake_y = self.G1.predict(img_x) #translate x to denoised x
                fake_Y = self.SR.predict(fake_y) #translate denoised x to super-resolved x
                
                # Train the discriminators (original images = real / translated = fake)
                #we will need different adversarial ground truths for the discriminators
                D1_loss_real = self.D1.train_on_batch(img_y, valid_D1)
                D1_loss_fake = self.D1.train_on_batch(fake_y, fake_D1)
                D1_loss = 0.5 * np.add(D1_loss_real, D1_loss_fake)
                
                D2_loss_real = self.D2.train_on_batch(img_Y, valid_D2)
                D2_loss_fake = self.D2.train_on_batch(fake_Y, fake_D2)
                D2_loss = 0.5 * np.add(D2_loss_real, D2_loss_fake)
                
                # ------------------
                #  Train Generators
                # ------------------
                
                blur_img_x = self.blur.predict(img_x) #passes img_x through the blurring kernel to provide GT.
                
                # Train the combined model (all generators)
                g_loss = self.combined.train_on_batch([img_x, img_y, img_Y], 
                                                      [valid_D1, img_x, blur_img_x, fake_y, 
                                                       valid_D2, img_x, fake_Y])
                
                """update log values"""
                #save the training point (measured in epochs)
                self.training.append(round(epoch+batch/self.data_loader.n_batches, 3))
                #adversarial losses
                self.D1_loss.append(D1_loss[0])
                self.D2_loss.append(D2_loss[0])
                self.G1_adv.append(g_loss[1])
                self.SR_adv.append(g_loss[5])
                
                #1cycleGAN losses
                self.cyc1.append(g_loss[2])
                self.blur1.append(g_loss[3])
                self.tv1.append(g_loss[4])
                
                #2nd cycleGan losses
                self.cyc2.append(g_loss[6])
                self.tv2.append(g_loss[7])
        
                print("[Epoch %d/%d] [Batch %d/%d]--[D1_adv: %.3f] [D2_adv: %.3f] -- [G1_adv: %.3f] [SR_adv: %.3f] [cyc1: %.4f] [cyc2: %.4f]" % (epoch, epochs,
                      batch, self.data_loader.n_batches, D1_loss[0], D2_loss[0], g_loss[1], g_loss[5], g_loss[2], g_loss[6]))
                
                if batch % 20 == 0 and batch!=0:
                    """save the model"""
                    model_name="{}_{}.h5".format(epoch, batch)
                    self.SR.save("models/"+model_name)
                    print("Epoch: {} --- Batch: {} ---- saved".format(epoch, batch))
                    
                    dynamic_evaluator.model = [self.G1, self.SR]
                    dynamic_evaluator.epoch = epoch
                    dynamic_evaluator.batch = batch
                    dynamic_evaluator.perceptual_test(5)
                    
                    sample_mean_ssim = dynamic_evaluator.objective_test(batch_size=250)
                    print("Sample mean SSIM: -------------------  %05f   -------------------" % (sample_mean_ssim))
                    self.ssim_eval_time.append(round(epoch+batch/self.data_loader.n_batches, 3))
                    self.ssim.append(sample_mean_ssim)
                    
                    self.log()
                    
                #elapsed_time = time.time() - start_time
                
        


patch_size=(25,25,3)
epochs=5
batch_size=6
sample_interval = 50 #after sample_interval batches save the model and generate sample images

gan = CINGAN(img_shape=patch_size)
gan.train(epochs=epochs, batch_size=batch_size, sample_interval=sample_interval)
