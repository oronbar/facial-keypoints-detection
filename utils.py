import pandas as pd
import cv2 
import numpy as np
import matplotlib.pyplot as plt


def get_train_ds():
    train_ds = pd.read_csv(r'training\training.csv')
    processed_images = []
    for (i,x) in train_ds.iterrows():
        processed_images.append(np.fromstring(x['Image'], sep=' ').astype(np.uint8).reshape(96,96))
    train_ds['Image'] = processed_images
    
    return train_ds

def get_test_ds():
    test_ds = pd.read_csv(r'test\test.csv', usecols=range(0,1))
    return test_ds

def get_image(n):
    image_df = pd.read_csv(r'training\training.csv', skiprows=lambda x: x != 0 and x != n)
    image = np.fromstring(image_df['Image'].iloc[0], sep=' ').astype(np.uint8).reshape(96,96)
    return image

def show_image_cv2(image):
    cv2.imshow('Image',image)
    cv2.waitKey(0)

def show_image(image):
    plt.imshow(image, cmap='gray')
    plt.show()

def main():
    train_ds = get_train_ds()
    show_image(get_image(5))

if __name__ == "__main__":
    main()