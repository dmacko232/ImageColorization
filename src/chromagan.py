import tensorflow as tf
import skimage
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_learning_curves, depreprocess, reconstruct
from loss import (
    wasserstein_generator_loss, wasserstein_discriminator_loss, 
    GradientPenaltyLoss, RandomWeightedAverage)
from tqdm.notebook import tqdm

class ChromaGAN:
    
    def __init__(self, image_shape=(224, 224), classes=20, vgg16_without_top=None):
        
        # in constructor we need to take generator, discriminator, join them up and setup any additional stuff
        
        # specify image shapes
        self.image_shape = image_shape
        self.input_ab_shape = (*image_shape, 2)
        self.input_L_shape = (*image_shape, 1)
        self.generator_input_shape = (*image_shape, 3)
        
        # setup image inputs from parts of network
        # a is geometric color information
        # b is semantic color information
        # L gray scale color image
        input_ab_real = tf.keras.Input(shape=self.input_ab_shape)
        input_L = tf.keras.Input(shape=self.input_L_shape)
        # three times the grayscale img, pretty much 3xL
        input_generator = tf.keras.Input(shape=self.generator_input_shape) 
        
        # setup discriminator
        self.discriminator = self.__class__.create_discriminator(
            self.input_ab_shape, self.input_L_shape)
        self.discriminator.compile(
            loss=wasserstein_discriminator_loss, 
            optimizer=tf.keras.optimizers.Adam(0.00002, 0.5)
        )
        
        # setup generator
        self.vgg16 = self.__class__.create_vgg16(self.generator_input_shape) if vgg16_without_top is None else vgg16_without_top
        self.generator = self.__class__.create_generator(self.generator_input_shape, self.vgg16, classes=classes)
        self.generator.compile(loss=['kld', 'mse'], optimizer=tf.keras.optimizers.Adam(0.00002, 0.9))

        # setup image outputs that join together different logical parts of discriminator and generator
        self.generator.trainable = False
        output_ab_generated, output_class_generated = self.generator(input_generator)
        output_discriminator_real = self.discriminator([input_ab_real, input_L]) 
        output_discriminator_generated = self.discriminator([output_ab_generated, input_L])
               
        # create layer to average real ab and ab prediction
        ab_averaged_real_generated = RandomWeightedAverage()([input_ab_real, output_ab_generated])
        output_averaged_real_generated = self.discriminator([ab_averaged_real_generated, input_L])
        
        # join up discriminator model with all the inputs/outputs
        # this is done because we need it to attach to many different parts
        self.discriminator_model = tf.keras.Model(
            inputs=[input_L, input_ab_real, input_generator],
            outputs=[output_discriminator_real, output_discriminator_generated, output_averaged_real_generated]
        )
        self.discriminator_model.compile(
            optimizer=tf.keras.optimizers.Adam(0.00002, 0.5),
            loss=[wasserstein_discriminator_loss, wasserstein_discriminator_loss, GradientPenaltyLoss(ab_averaged_real_generated)],
            loss_weights=[-1.0, 1.0, 1.0] # we use wasserstein once with -1 coeff and once with 1
        )
        
        # create network from both
        self.generator.trainable = True
        self.discriminator.trainable = False
        self.network = tf.keras.Model(
            inputs=[input_generator, input_L],
            outputs=[output_ab_generated, output_class_generated, output_discriminator_generated]
        )
        self.network.compile(
            optimizer=tf.keras.optimizers.Adam(0.00002, 0.5),
            loss=['mse', 'kld', wasserstein_generator_loss],
            loss_weights=[1.0, .003, -0.1],
        )
        
        # set up loss arrays
        self.generator_loss = [] # actually loss for whole combined model
        self.discriminator_loss = []
        
    def train_one_epoch(self, train_dataset, valid, fake, dummy):
        
        for batch_i in tqdm(range(len(train_dataset))):
            
            gray_imgs, color_imgs, gray_lab_imgs, color_lab_imgs, encoded_labels, labels = train_dataset[batch_i]
            train_L = gray_lab_imgs
            train_ab = color_lab_imgs
            # we stack the L (gray color) three times
            train_L3 = np.tile(train_L, [1, 1, 1, 3])
            
            # don't use pretrained predictor but use labels
            
            generator_loss = self.network.train_on_batch(
                [train_L3, train_L], 
                [train_ab, encoded_labels, valid]
            )
            self.generator_loss.append(generator_loss)
            
            discriminator_loss = self.discriminator_model.train_on_batch(
                [train_L, train_ab, train_L3],
                [valid, fake, dummy]
            )
            self.discriminator_loss.append(discriminator_loss)
        
        
    def train(self, train_dataset, test_dataset, epochs=30, save=True, display_test_image_results=True):
        
        # create dummy, fake, and real values for discriminator to train
        valid = np.ones((train_dataset.batch_size, 1))
        fake = - valid
        dummy = np.zeros((train_dataset.batch_size, 1)) # for gradient penalty
        
        for e in tqdm(range(epochs)):
            
            self.train_one_epoch(train_dataset, valid, fake, dummy)
            
            # each few epochs plot and show images
            if (e % 5 == 0) or e == epochs-1: 

                if display_test_image_results:
                    self.display_colorized_samples(test_dataset)
                plot_learning_curves(self.discriminator_loss, self.generator_loss)
                
                if save:
                    self.generator.save(f"generator_epoch{e}")
                    self.discriminator.save(f"discriminator_epoch{e}")
                    self.network.save(f"network_epoch{e}")
            
            train_dataset.on_epoch_end()
                                 
                            
    def display_colorized_samples(self, test_dataset, batch_i=0):
        
        gray_imgs, color_imgs, gray_lab_imgs, color_lab_imgs, encoded_labels, labels = test_dataset[batch_i]
        colorized = self.__class__.colorize(self.generator, gray_lab_imgs)
        for i in range(len(colorized)):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(12, 6))
            ax1.imshow(gray_lab_imgs[i][:, :, 0], cmap="gray")
            ax1.set_title(f"Gray for {labels[i]}")
            ax2.imshow(colorized[i])
            ax2.set_title(f"Colorized")
            ax3.imshow(color_imgs[i],)
            ax3.set_title(f"Ground truth ({labels[i]})")
            plt.show()
    
    @classmethod
    def colorize(cls, generator, imgs_L):
        
        imgs_ab, _ = generator.predict(np.tile(imgs_L, [1, 1, 1, 3]))
        # we NEED to depreprocess BEFORE reconstruction (joining them back) because
        # we scaled down the images in data loader by dividing..
        # and to make sense before the conversion they need to be scaled up 
        # WITHOUT THIS STEP THE RESULT IMGS DONT MAKE _ANY_ SENSE
        return reconstruct(depreprocess(imgs_L), depreprocess(imgs_ab))
        
    @staticmethod
    def create_discriminator(input_ab_shape, input_L_shape):
        
        # a is geometric color information
        # b is semantic color information
        # L is gray scale color image
        input_ab = tf.keras.Input(shape=input_ab_shape, name='discriminator_geometric/semantic_input')
        input_L = tf.keras.Input(shape=input_L_shape, name='discriminator_grayscale_input')
        discriminator = tf.keras.layers.concatenate([input_L, input_ab])
        
        # TODO maybe add batch normalization but not for first layer
        
        discriminator = tf.keras.layers.Conv2D(
            64, (4, 4), padding='same', strides=(2, 2), activation=tf.keras.layers.LeakyReLU(alpha=0.2)
        )(discriminator) # (112, 112, 64)
        
        discriminator = tf.keras.layers.Conv2D(
            128, (4, 4), padding='same', strides=(2, 2), activation=tf.keras.layers.LeakyReLU(alpha=0.2)
        )(discriminator) # (56, 56, 128)
        
        discriminator = tf.keras.layers.Conv2D(
            256, (4, 4), padding='same', strides=(2, 2), activation=tf.keras.layers.LeakyReLU(alpha=0.2)
        )(discriminator) # (28, 28, 256)
        
        discriminator = tf.keras.layers.Conv2D(
            512, (4, 4), padding='same', strides=(1, 1), activation=tf.keras.layers.LeakyReLU(alpha=0.2)
        )(discriminator) # (28, 28, 512)
        
        # TODO maybe rewrite
        discriminator = tf.keras.layers.Conv2D(
            1, (4, 4), padding='same', strides=(1, 1), activation='sigmoid'
        )(discriminator) # (28, 28, 1)
        
        return tf.keras.Model(inputs=[input_ab, input_L], outputs=discriminator)
        
    @staticmethod
    def create_generator(input_shape, vgg16_without_top, classes=20):
        
        input_layer = tf.keras.Input(shape=input_shape)
        # take only subset of the layers of vgg16
        vgg16_model = tf.keras.Model(vgg16_without_top.input, vgg16_without_top.layers[-6].output)
        # join up the input with vgg16 model -- this is the YELLOW in the image
        model = vgg16_model(input_layer)
        
        # global features are the lower RED level in the image
        # the first 4 red ones from left, they get input from vgg16
        global_features1 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', strides=(2, 2), activation='relu')(model)
        global_features1 = tf.keras.layers.BatchNormalization()(global_features1)
        
        global_features2 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu')(global_features1)
        global_features2 = tf.keras.layers.BatchNormalization()(global_features2)
        
        global_features3 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', strides=(2, 2), activation='relu')(global_features2)
        global_features3 = tf.keras.layers.BatchNormalization()(global_features3)
        
        global_features4 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu')(global_features3)
        global_features4 = tf.keras.layers.BatchNormalization()(global_features4)
        
        # global features class, GRAY in the image! classifies the image as one of the classes
        global_features_class = tf.keras.layers.Flatten()(global_features4)
        global_features_class = tf.keras.layers.Dense(4096)(global_features_class)
        global_features_class = tf.keras.layers.Dense(4096)(global_features_class)
        global_features_class = tf.keras.layers.Dense(classes, activation='softmax')(global_features_class)
        
        # this connects the global features to the model result, RED
        global_features_connector = tf.keras.layers.Flatten()(global_features2)
        global_features_connector = tf.keras.layers.Dense(1024)(global_features_connector)
        global_features_connector = tf.keras.layers.Dense(512)(global_features_connector)
        global_features_connector = tf.keras.layers.Dense(256)(global_features_connector)
        global_features_connector = tf.keras.layers.RepeatVector(28 * 28)(global_features_connector)
        global_features_connector = tf.keras.layers.Reshape((28, 28, 256))(global_features_connector)
        
        # mid features, PURPLE in image
        mid_features = tf.keras.layers.Conv2D(512, (3, 3),  padding='same', strides=(1, 1), activation='relu')(model)
        mid_features = tf.keras.layers.BatchNormalization()(mid_features)
        mid_features = tf.keras.layers.Conv2D(256, (3, 3),  padding='same', strides=(1, 1), activation='relu')(mid_features)
        mid_features = tf.keras.layers.BatchNormalization()(mid_features)
        
        # colorization output, BLUE in image
        colorization_output = tf.keras.layers.concatenate([mid_features, global_features_connector])
        colorization_output = tf.keras.layers.Conv2D(256, (1, 1), padding='same', strides=(1, 1), activation='relu')(colorization_output)
        colorization_output = tf.keras.layers.Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu')(colorization_output)

        colorization_output = tf.keras.layers.UpSampling2D(size=(2,2))(colorization_output)
        colorization_output = tf.keras.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(colorization_output)
        colorization_output = tf.keras.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(colorization_output)

        colorization_output = tf.keras.layers.UpSampling2D(size=(2, 2))(colorization_output)
        colorization_output = tf.keras.layers.Conv2D(32, (3, 3), padding='same', strides=(1, 1), activation='relu')(colorization_output)
        colorization_output = tf.keras.layers.Conv2D(2, (3, 3), padding='same', strides=(1, 1), activation='sigmoid')(colorization_output)
        colorization_output = tf.keras.layers.UpSampling2D(size=(2, 2))(colorization_output)
        
        # output is (a, b)
        # a is geometric color information
        # b is semantic color information
        return tf.keras.Model(
            inputs=input_layer, 
            outputs=[colorization_output, global_features_class], 
            name="generator"
        )
                            
        
    @classmethod
    def create_vgg16(cls, input_shape, include_top=False):
        
        # pretrained vgg16 on imagenet usable as backbone or predictor for generator!
        # it is also possible to load it, further finetune it on other dataset (natural color)
        # and then use it to get generator
        return tf.keras.applications.vgg16.VGG16(
            weights='imagenet', include_top=include_top, input_shape=input_shape)
