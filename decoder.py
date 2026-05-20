import cv2
import os
import numpy as np
from Globale.convertionCouleur import postprocess_frame
from Globale.DCT               import decoder_iframe
from Globale.ompensation       import decoder_pframe
from Globale.entropique        import load_and_decompress
import configuration as cfg


def decode_all(chemin_bin, dossier_sortie="Sortie/decoded_frames"):
    print("\n--- decodeur ---")
    frames_enc = load_and_decompress(chemin_bin)
    
    frames_rec = []
    frames_bgr = []
    ref = None

    for idx, enc in enumerate(frames_enc):
        if enc['type'] == 'I':
            rec = decoder_iframe(enc)
        else:
            rec = decoder_pframe(enc, ref)
        frames_rec.append(rec)
        ref = rec
        bgr = postprocess_frame(rec['Y'], rec['Cb'], rec['Cr'])
        frames_bgr.append(bgr)
        print(f"  frame {idx+1:02d} ({enc['type']}) decodee")

    os.makedirs(dossier_sortie, exist_ok=True)
    for i, f in enumerate(frames_bgr):
        cv2.imwrite(os.path.join(dossier_sortie, f"frame_{i+1:03d}.png"), f)
    print(f"  frames sauvegardees dans {dossier_sortie}")
    return frames_bgr