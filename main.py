import argparse
from encoder import VideoEncoder
import configuration as config
from Globale.affichage import visualize_dct_block
def main():
    parser = argparse.ArgumentParser(description="Codec vidéo MPEG-4 simplifié")
    parser.add_argument('--mode', choices=['encode', 'decode'], default='encode',
                        help='Mode d\'exécution')
    
    args = parser.parse_args()
    
    if args.mode == 'encode':
        print("=== ENCODEUR VIDÉO MPEG-4 simplifié ===\n")
        encoder = VideoEncoder()
        processed_frames = encoder.run(config.INPUT_FRAMES_DIR)
        
        print("\n💡  debut  Partie 2 ")

        sample = processed_frames[0]  # UNE seule frame
        visualize_dct_block(sample['Y'], qf=50)
        
    else:
        print("Mode décodeur : à implémenter plus tard")
        # decoder = VideoDecoder()
        # decoder.run(config.OUTPUT_BIN)

if __name__ == "__main__":
    main()