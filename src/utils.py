import matplotlib.pyplot as plt
import skimage.color
import numpy as np

def plot_learning_curves(d_loss, g_loss):
    """Plot learning curves of discriminator and generator."""

    plt.plot(d_loss)
    plt.plot(g_loss)
    plt.ylim(ymax=np.max([d_loss, g_loss]), ymin=0)
    plt.legend(['Discriminator loss', 'Generator loss'])
    plt.title('Learning curves')
    plt.show()
    
def depreprocess(imgs):
    """Depreprocess image."""
    
    imgs *= 255
    imgs[imgs > 255] = 255
    imgs[imgs < 0] = 0
    return imgs

def reconstruct(gray_lab_imgs, color_lab_imgs):
    """Reconstruct image converting from lab color space to rgb."""
    
    return skimage.color.lab2rgb(np.concatenate((gray_lab_imgs, color_lab_imgs), axis=3))
    
