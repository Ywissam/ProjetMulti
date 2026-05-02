# utils/visualize.py
import cv2
from matplotlib.pylab import block
import matplotlib.pyplot as plt
import numpy as np
from Globale.DCT import dct2d, idct2d, get_quantization_matrix
"partie 1"
def visualize_preprocessing(original_bgr, Y, Cb, Cr, title="Prétraitement"):
    """
    Visualisation des canaux Y, Cb, Cr comme demandé dans l'énoncé.
    
    Utilisation : À appeler pendant le débogage ou pour la partie 5b.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Afficher l'originale (BGR → RGB pour matplotlib)
    axes[0].imshow(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Originale (RGB)")
    axes[0].axis('off')
    
    # Afficher Y (luminance)
    axes[1].imshow(Y, cmap='gray')
    axes[1].set_title("Canal Y (Luminance)")
    axes[1].axis('off')
    
    # Afficher Cb
    axes[2].imshow(Cb, cmap='gray')
    axes[2].set_title("Canal Cb (Chrominance bleue)")
    axes[2].axis('off')
    
    # Afficher Cr
    axes[3].imshow(Cr, cmap='gray')
    axes[3].set_title("Canal Cr (Chrominance rouge)")
    axes[3].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def visualize_subsampling(cb_full, cb_subsampled):
    """
    Visualisation de l'effet du sous-échantillonnage.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(cb_full, cmap='gray')
    axes[0].set_title(f"Cb original ({cb_full.shape[1]}x{cb_full.shape[0]})")
    axes[0].axis('off')
    
    axes[1].imshow(cb_subsampled, cmap='gray')
    axes[1].set_title(f"Cb sous-échantillonné ({cb_subsampled.shape[1]}x{cb_subsampled.shape[0]})")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

    def visualize_dct_block(Y, qf=50):
   

    # choisir un bloc (évite coin noir)
     block = Y[32:40, 32:40].astype(np.float32)

    Q = get_quantization_matrix('luma', qf) # type: ignore

    # pipeline
    centered = np.block - 128
    dct_block = dct2d(centered)
    quant_block = np.round(dct_block / Q)
    dequant = quant_block * Q
    recon_block = idct2d(dequant) + 128

    # affichage
    plt.figure(figsize=(10,3))

    plt.subplot(1,4,1)
    plt.imshow(block, cmap='gray')
    plt.title("Original 8x8")
    plt.axis('off')

    plt.subplot(1,4,2)
    plt.imshow(np.log(np.abs(dct_block)+1), cmap='gray')
    plt.title("DCT")
    plt.axis('off')

    plt.subplot(1,4,3)
    plt.imshow(quant_block, cmap='gray')
    plt.title("Quantized")
    plt.axis('off')

    plt.subplot(1,4,4)
    plt.imshow(recon_block, cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

    plt.tight_layout()
    plt.show()