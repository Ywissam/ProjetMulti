# config.py
# Tous les paramètres du projet centralisés ici

# Paramètres vidéo
FRAME_WIDTH = 640   # Largeur des frames (à adapter)
FRAME_HEIGHT = 480  # Hauteur des frames (à adapter)

# Paramètres de prétraitement
COLOR_SPACE = 'YCbCr'  # On convertit BGR → YCbCr
CHROMA_SUBSAMPLING = '4:2:0'  # Format de sous-échantillonnage

# Chemins
INPUT_FRAMES_DIR = "frames"   # Dossier avec les frames extraites
OUTPUT_BIN = "Sortie/compressed.bin"  # Fichier compressé