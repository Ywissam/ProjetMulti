import cv2
import os
import numpy as np
from Globale.convertionCouleur import preprocess_frame
from Globale.DCT               import encoder_iframe, decoder_iframe
from Globale.ompensation       import encoder_pframe, decoder_pframe
from Globale.entropique        import compress_and_save, compute_compression_ratio
import configuration as cfg


def load_frames(dossier):
    fichiers = sorted([f for f in os.listdir(dossier)
                       if f.lower().endswith(('.png','.jpg'))])
    print(f"chargement de {len(fichiers)} frames...")
    frames = []
    for f in fichiers:
        img = cv2.imread(os.path.join(dossier, f))
        if img is None:
            continue
        if img.shape[1] != cfg.FRAME_WIDTH or img.shape[0] != cfg.FRAME_HEIGHT:
            img = cv2.resize(img, (cfg.FRAME_WIDTH, cfg.FRAME_HEIGHT))
        frames.append(img)
    print(f"  {len(frames)} frames chargees")
    return frames


def preprocess_all(frames):
    print("\n--- partie 1 : pretraitement ---")
    result = []
    for f in frames:
        Y, Cb, Cr = preprocess_frame(f)
        result.append({'Y': Y, 'Cb': Cb, 'Cr': Cr})
    s = result[0]
    print(f"  Y:{s['Y'].shape}  Cb:{s['Cb'].shape}  Cr:{s['Cr'].shape}")
    return result


def encode_all(frames_pre):
    print(f"\n--- parties 2 et 3 : encodage (GOP={cfg.GOP_SIZE} QF={cfg.QF}) ---")
    frames_enc = []
    frames_rec = []
    ni = np_ = 0

    for idx, pf in enumerate(frames_pre):
        if idx % cfg.GOP_SIZE == 0:
            enc = encoder_iframe(pf, qf=cfg.QF)
            rec = decoder_iframe(enc)
            ni += 1
            print(f"  frame {idx+1:02d} -> I-frame")
        else:
            enc = encoder_pframe(pf, frames_rec[-1],
                                 qf=cfg.QF, fenetre=cfg.SEARCH_RANGE)
            rec = decoder_pframe(enc, frames_rec[-1])
            np_ += 1
            print(f"  frame {idx+1:02d} -> P-frame")
        frames_enc.append(enc)
        frames_rec.append(rec)

    print(f"  {ni} I-frames  {np_} P-frames")
    return frames_enc, frames_rec


def save_bin(frames_orig, frames_enc, chemin):
    taille_comp = compress_and_save(frames_enc, chemin)
    ratio, orig = compute_compression_ratio(frames_orig, taille_comp)
    print(f"  original   : {orig/1e6:.2f} Mo")
    print(f"  compresse  : {taille_comp/1e6:.2f} Mo")
    print(f"  ratio      : {ratio:.2f}x")
    return ratio