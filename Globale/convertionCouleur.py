# utils/color_conversion.py
import numpy as np
import cv2

"partie 1"

def bgr_to_ycbcr(image_bgr):
    """
    Convertit une image BGR (0-255) en YCbCr.
    
    Justification : On utilise les formules BT.601 standard
    car elles sont utilisées dans MPEG-4 et JPEG.
    
    Args:
        image_bgr: numpy array (H, W, 3) en BGR
    
    Returns:
        Y: canal luminance (H, W)
        Cb: canal chrominance bleu (H, W)
        Cr: canal chrominance rouge (H, W)
    """
    # Normaliser les valeurs entre 0 et 1
    img = image_bgr.astype(np.float32) / 255.0
    
    # Extraire les canaux BGR (OpenCV lit en BGR)
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]
    
    # Formules BT.601
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 128.0 / 255.0 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 128.0 / 255.0 + 0.5 * R - 0.418688 * G - 0.081312 * B
    
    # Remettre dans l'échelle 0-255
    Y = (Y * 255).astype(np.uint8)
    Cb = (Cb * 255).astype(np.uint8)
    Cr = (Cr * 255).astype(np.uint8)
    
    return Y, Cb, Cr

def ycbcr_to_bgr(Y, Cb, Cr):
    """
    Convertit YCbCr → BGR (inverse de la fonction ci-dessus).
    Utile pour le décodeur.
    """
    # Normaliser entre 0 et 1
    Y = Y.astype(np.float32) / 255.0
    Cb = Cb.astype(np.float32) / 255.0
    Cr = Cr.astype(np.float32) / 255.0
    
    # Formules inverses BT.601
    R = Y + 1.402 * (Cr - 128.0/255.0)
    G = Y - 0.344136 * (Cb - 128.0/255.0) - 0.714136 * (Cr - 128.0/255.0)
    B = Y + 1.772 * (Cb - 128.0/255.0)
    
    # Remettre dans l'échelle 0-255 et clamp
    R = np.clip(R * 255, 0, 255).astype(np.uint8)
    G = np.clip(G * 255, 0, 255).astype(np.uint8)
    B = np.clip(B * 255, 0, 255).astype(np.uint8)
    
    # Reconstruire l'image BGR
    bgr_image = np.stack([B, G, R], axis=2)
    
    return bgr_image

def chroma_subsampling_420(cb, cr):
    """
    Sous-échantillonnage 4:2:0.
    On réduit Cb et Cr par un facteur 2 en hauteur et largeur.
    
    Justification : L'œil humain est moins sensible aux détails de couleur.
    Cette réduction divise par 2 la taille des canaux chrominance.
    
    Args:
        cb: canal Cb (H, W)
        cr: canal Cr (H, W)
    
    Returns:
        cb_subsampled: (H//2, W//2)
        cr_subsampled: (H//2, W//2)
    """
    # Prendre la moyenne sur des blocs 2x2
    h, w = cb.shape
    # S'assurer que les dimensions sont paires
    h_even = h - (h % 2)
    w_even = w - (w % 2)
    
    cb_trimmed = cb[:h_even, :w_even]
    cr_trimmed = cr[:h_even, :w_even]
    
    # Redimensionner par moyenne (méthode plus propre que simple sous-échantillonnage)
    cb_subsampled = cv2.resize(cb_trimmed, (w_even//2, h_even//2), interpolation=cv2.INTER_AREA)
    cr_subsampled = cv2.resize(cr_trimmed, (w_even//2, h_even//2), interpolation=cv2.INTER_AREA)
    
    return cb_subsampled, cr_subsampled

def chroma_upsampling_420(cb_subsampled, cr_subsampled, original_shape):
    """
    Upsampling des canaux Cb et Cr (pour le décodeur).
    On revient à la taille originale.
    
    Args:
        cb_subsampled: (H//2, W//2)
        cr_subsampled: (H//2, W//2)
        original_shape: (H, W) taille originale des canaux Y
    
    Returns:
        cb_upsampled: (H, W)
        cr_upsampled: (H, W)
    """
    h, w = original_shape
    cb_upsampled = cv2.resize(cb_subsampled, (w, h), interpolation=cv2.INTER_LINEAR)
    cr_upsampled = cv2.resize(cr_subsampled, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return cb_upsampled, cr_upsampled

def preprocess_frame(frame_bgr):
    """
    Pipeline complet de prétraitement pour UNE frame.
    
    Args:
        frame_bgr: image BGR (H, W, 3)
    
    Returns:
        Y: (H, W)
        Cb_subsampled: (H//2, W//2)
        Cr_subsampled: (H//2, W//2)
    """
    # 1. Conversion BGR → YCbCr
    Y, Cb, Cr = bgr_to_ycbcr(frame_bgr)
    
    # 2. Sous-échantillonnage 4:2:0
    Cb_sub, Cr_sub = chroma_subsampling_420(Cb, Cr)
    
    return Y, Cb_sub, Cr_sub

def postprocess_frame(Y, Cb_sub, Cr_sub):
    """
    Pipeline inverse de post-traitement pour le décodeur.
    
    Args:
        Y: (H, W)
        Cb_sub: (H//2, W//2)
        Cr_sub: (H//2, W//2)
    
    Returns:
        frame_bgr: image BGR (H, W, 3)
    """
    # 1. Upsampling
    Cb_up, Cr_up = chroma_upsampling_420(Cb_sub, Cr_sub, Y.shape)
    
    # 2. Conversion YCbCr → BGR
    frame_bgr = ycbcr_to_bgr(Y, Cb_up, Cr_up)
    
    return frame_bgr