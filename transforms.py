"""
Modified from Zachary's notebook
"""
import numpy as np
import cv2
import random
import kornia
import torch

def get_random_gaussian(max_sigma=0.1):
    def add_gaussian_noise(batch):
        num, ch, row, col = batch.shape
        mean = 0
        sigma = np.random.uniform(0, max_sigma)
        gauss = np.random.normal(mean, sigma, (num, ch, row, col))
        #gauss = gauss.reshape(num, row, col, ch)
        noisy = np.clip(batch + gauss, 0., 1.)
        return noisy
    return add_gaussian_noise

def get_random_rotate(min_deg=-22.5, max_deg=22.5):
    def rotate_random(batch):
        num, ch, row, col = batch.shape
        rotated = []
        for img in batch:
            deg = np.random.randint(min_deg, max_deg)
            M = cv2.getRotationMatrix2D((col/2, row/2), deg, 1)
            dst = np.array([cv2.warpAffine(channel, M, (col, row)) for channel in img])
            rotated.append(dst)
        return np.array(rotated)
    return rotate_random

def get_random_contrast(min_alpha=0.5, max_alpha=1.5):
    def contrast_random(batch):
        adjusted = []
        for img in batch:
            alpha = np.random.uniform(min_alpha, max_alpha)
            dst = np.clip(alpha * img, 0., 1.)
            adjusted.append(dst)
        return np.array(adjusted)
    return contrast_random

def get_random_brightness(min_beta=-0.05, max_beta=0.5):
    def brightness_random(batch):
        adjusted = []
        for img in batch:
            beta = np.random.uniform(min_beta, max_beta)
            dst = np.clip(img + beta, 0., 1.)
            adjusted.append(dst)
        return np.array(adjusted)
    return brightness_random
  
def get_random_blur(min_blur_size=1, max_blur_size=10):
    def blur_random(batch):
        num, ch, row, col = batch.shape
        blurred = []
        for img in batch:
            blur_size = np.random.randint(min_blur_size / 2, max_blur_size / 2) * 2 + 1
            dst = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
            blurred.append(dst)
        return np.array(blurred)
    return blur_random

class RandomTransform:
    def __init__(self, T):
        self.T = T # List of candidate transformations
        
    def __call__(self, x):
        t = random.choice(self.T)
        return t(x)
    
def get_random_gaussian_pt(max_sigma=0.1):
    def add_gaussian_noise(x):
        mean = 0
        sigma = np.random.uniform(0, max_sigma)
        gauss = sigma*torch.randn_like(x).to(x.device)
        x_t = torch.clamp(x + gauss, 0., 1.)
        return x_t
    return add_gaussian_noise

def get_random_contrast_pt(min_alpha=0.9, max_alpha=1.4):
    def contrast_random(x):
        alpha = np.random.uniform(min_alpha, max_alpha)
#         alpha = torch.rand((x.size(0), 1, 1, 1)).to(x.device)
#         alpha = (alpha - max_alpha) / (max_alpha-min_alpha)
        x_t = torch.clamp(x*alpha, 0., 1.)
        return x_t
    return contrast_random

def get_random_brightness_pt(min_beta=-0.05, max_beta=0.05):
    def brightness_random(x):
        beta = np.random.uniform(min_beta, max_beta)
#         beta = torch.rand((x.size(0), 1, 1, 1)).to(x.device)
#         beta = (beta - max_beta) / (max_beta-min_beta)
        x_t = torch.clamp(x+beta, 0., 1.)
        return x_t
    return brightness_random

def get_random_rotate_kornia(max_deg=22.5):
    def rotate_random(x):
        rotate = kornia.augmentation.RandomRotation(max_deg, return_transform=False)
        return rotate(x)
    return rotate_random