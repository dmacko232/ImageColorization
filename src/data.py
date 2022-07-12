import tensorflow as tf
#import tensorflow_io as tfio
import pandas as pd
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import skimage.color
import skimage.transform
import skimage.io

def build_dataframe(colorful_dirpath: str, gray_dirpath: str) -> pd.DataFrame:
    
    result = []
    for label in os.listdir(gray_dirpath):
        for image_name in os.listdir(os.path.join(gray_dirpath, label)):
            gray_path = os.path.join(gray_dirpath, label, image_name)
            colorful_path = os.path.join(colorful_dirpath, label, image_name)
            if os.path.exists(colorful_path): # ignore images where we only have the gray one
                result.append((gray_path, colorful_path, label))
            
            
    return pd.DataFrame(result, columns=["gray_path", "colorful_path", "label"])

class DataLoader(tf.keras.utils.Sequence):
    
    def __init__(self,
                 df: pd.DataFrame,
                 dim=(224, 224),
                 batch_size=16,
                 training=True,
                 seed=0
                ):
    
        self.df = df                           # description of input samples
        self.dim = dim                         # input image size
        self.batch_size = batch_size           # batch size
        self.training = training               # True to augment and shuffle samples  
        self.n_classes = self.df["label"].nunique()
        self.label_indices = DataLoader._build_label_indices(self.df["label"])
        self.n = len(self.df)
        
        # set seed
        self.seed = seed
        np.random.seed(self.seed)
        
        # encode clases to array
        self.encoded_classes = tf.one_hot(
            self.label_indices.to_numpy(), 
            depth=self.n_classes, 
        )
        
        self.on_epoch_end()

    def __len__(self):
        ' get number of batches in one epoch '
        
        return self.n // self.batch_size
        
    def __getitem__(self, index):
        
        index_start = index * self.batch_size
        index_end = (index + 1) * self.batch_size
        encoded_labels = self.encoded_classes[index_start: index_end]
        batch_df = self.df[index_start: index_end]
        labels = batch_df["label"].to_numpy()
        gray_imgs = np.stack([
            self._load_gray_img(path) 
            for path 
            in batch_df["gray_path"]
        ])
        colorful_imgs = [
            self._load_color_img(path) 
            for path 
            in batch_df["colorful_path"]
        ]
        color_imgs = np.stack([t[0] for t in colorful_imgs])
        gray_lab_imgs = np.stack([t[1] for t in colorful_imgs])
        color_lab_imgs = np.stack([t[2] for t in colorful_imgs])
        return gray_imgs, color_imgs, gray_lab_imgs, color_lab_imgs, encoded_labels, labels
    
    def on_epoch_end(self):
        ' called on epoch end '
        
        # shuffle
        if not self.training:
            self.df = self.df.sample(frac=1.0, random_state=self.seed)
            
    def _load_gray_img(self, img_path: str):
        
        return np.reshape(skimage.transform.resize(skimage.io.imread(img_path), self.dim), (*self.dim, 1) )
    
    def _load_color_img(self, img_path: str):
        
        color_img = skimage.transform.resize(skimage.io.imread(img_path), self.dim)
        # TODO apply augmentations
        #if self.training:
            
        lab_img = skimage.color.rgb2lab(color_img) / 255 # converting scaled rgb to lab leads to unscaled lab
        #lab_img = tfio.experimental.color.rgb_to_lab(color_img)
        gray_lab_img = lab_img[:, :, 0:1] # 0:1 so we have (224, 224, 1) and not (224, 224)
        color_lab_img = lab_img[:, :, 1:]
        return color_img, gray_lab_img, color_lab_img
        
    @staticmethod
    def _build_label_indices(labels: pd.Series) -> pd.Series:
        
        unique = sorted(np.unique(labels))
        label2index = { l: i for i, l in enumerate(unique) }
        return labels.apply(lambda l: label2index[l])
