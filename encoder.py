# encoder.py
import cv2
import os
import numpy as np
from Globale.convertionCouleur import preprocess_frame
import configuration as config
from Globale.DCT import encode_iframe

"partie 1"
class VideoEncoder:
    def __init__(self):
        self.frames = []          # Stocke les frames brutes lues
        self.processed_frames = [] # Stocke les frames après prétraitement
        
    def load_frames(self, input_dir):
        """
        Charge toutes les images d'un dossier.
        """
        frame_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg'))])
        
        print(f"Chargement de {len(frame_files)} frames...")
        
        for file in frame_files:
            path = os.path.join(input_dir, file)
            frame = cv2.imread(path)  # Lecture en BGR
            
            if frame is None:
                print(f"Erreur : impossible de lire {file}")
                continue
                
            # Redimensionner si nécessaire (optionnel)
            if frame.shape[1] != config.FRAME_WIDTH or frame.shape[0] != config.FRAME_HEIGHT:
                frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
            
            self.frames.append(frame)
        
        print(f"✓ {len(self.frames)} frames chargées avec succès")
        return self.frames
    
    def preprocess_all_frames(self):
        """
        Applique le prétraitement (Partie 1) à toutes les frames.
        """
        print("\n--- Partie 1 : Prétraitement ---")
        print("Conversion BGR → YCbCr et sous-échantillonnage 4:2:0...")
        
        for i, frame in enumerate(self.frames):
            Y, Cb_sub, Cr_sub = preprocess_frame(frame)
            
            self.processed_frames.append({
                'Y': Y,
                'Cb': Cb_sub,
                'Cr': Cr_sub,
                'original': frame
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Traitement : {i+1}/{len(self.frames)} frames")
        
        print(f"✓ Prétraitement terminé sur {len(self.processed_frames)} frames")
        
        # Afficher les tailles pour vérification
        sample = self.processed_frames[0]
        print(f"\nTailles après prétraitement :")
        print(f"  Y : {sample['Y'].shape}")
        print(f"  Cb sous-échantillonné : {sample['Cb'].shape}")
        print(f"  Cr sous-échantillonné : {sample['Cr'].shape}")
        
        return self.processed_frames
    
    def run(self, input_dir):
        """
        Exécute l'encodeur.
        """
        self.load_frames(input_dir)
        self.preprocess_all_frames()
        
        print("\n✅ Partie 1 terminée avec succès !")
        print("Prochaine étape : Partie 2 (I-frames avec DCT)")
        self.encoded_frames = []

        for frame in self.processed_frames:
          encoded = encode_iframe(frame, qf=50)
        self.encoded_frames.append(encoded)

        return self.processed_frames

# Test rapide
if __name__ == "__main__":
    encoder = VideoEncoder()
    processed = encoder.run(config.INPUT_FRAMES_DIR)