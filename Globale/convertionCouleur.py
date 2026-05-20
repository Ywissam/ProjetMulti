import numpy as np
import cv2

def bgr_to_ycbcr(img):

    img_float = img.astype(np.float32) / 255.0
    
    B = img_float[:, :, 0]
    G = img_float[:, :, 1]
    R = img_float[:, :, 2]
    
    Y  =  0.299*R + 0.587*G + 0.114*B
    Cb = -0.168736*R - 0.331264*G + 0.5*B + (128.0/255.0)
    Cr =  0.5*R - 0.418688*G - 0.081312*B + (128.0/255.0)
    
    Y  = (Y  * 255).astype(np.uint8)
    Cb = (Cb * 255).astype(np.uint8)
    Cr = (Cr * 255).astype(np.uint8)
    
    return Y, Cb, Cr


def ycbcr_to_bgr(Y, Cb, Cr):
    Y  = Y.astype(np.float32)  / 255.0
    Cb = Cb.astype(np.float32) / 255.0
    Cr = Cr.astype(np.float32) / 255.0
    
    R = Y + 1.402    * (Cr - 128.0/255.0)
    G = Y - 0.344136 * (Cb - 128.0/255.0) - 0.714136*(Cr - 128.0/255.0)
    B = Y + 1.772    * (Cb - 128.0/255.0)
    
    R = np.clip(R * 255, 0, 255).astype(np.uint8)
    G = np.clip(G * 255, 0, 255).astype(np.uint8)
    B = np.clip(B * 255, 0, 255).astype(np.uint8)
    
    return np.stack([B, G, R], axis=2)


def sous_echantillonnage_420(Cb, Cr):

    h, w = Cb.shape
    h2 = h - (h % 2)
    w2 = w - (w % 2)
    Cb2 = cv2.resize(Cb[:h2, :w2], (w2//2, h2//2), interpolation=cv2.INTER_AREA)
    Cr2 = cv2.resize(Cr[:h2, :w2], (w2//2, h2//2), interpolation=cv2.INTER_AREA)
    return Cb2, Cr2


def upsampling_420(Cb_sub, Cr_sub, shape):
    # on remet Cb et Cr a la taille originale
    h, w = shape
    Cb_up = cv2.resize(Cb_sub, (w, h), interpolation=cv2.INTER_LINEAR)
    Cr_up = cv2.resize(Cr_sub, (w, h), interpolation=cv2.INTER_LINEAR)
    return Cb_up, Cr_up


def preprocess_frame(frame):
    Y, Cb, Cr = bgr_to_ycbcr(frame)
    Cb_sub, Cr_sub = sous_echantillonnage_420(Cb, Cr)
    return Y, Cb_sub, Cr_sub


def postprocess_frame(Y, Cb_sub, Cr_sub):
    Cb_up, Cr_up = upsampling_420(Cb_sub, Cr_sub, Y.shape)
    return ycbcr_to_bgr(Y, Cb_up, Cr_up)