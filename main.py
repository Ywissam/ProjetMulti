import argparse
from encoder import load_frames, preprocess_all, encode_all, save_bin
from decoder import decode_all
from Globale.affichage import visualize_pipeline, frame_type_breakdown
import configuration as cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['encode','decode','all'], default='all')
    args = parser.parse_args()

    if args.mode in ('encode', 'all'):
        print("="*45)
        print("  encodeur MPEG-4 simplifie")
        print("="*45)

        frames_orig = load_frames(cfg.INPUT_FRAMES_DIR)
        frames_pre  = preprocess_all(frames_orig)
        frames_enc, frames_rec = encode_all(frames_pre)

        print("\nvisualisation du pipeline...")
        visualize_pipeline(frames_orig, frames_pre, frames_enc, frames_rec)

        ratio = save_bin(frames_orig, frames_enc, cfg.OUTPUT_BIN)
        ni, np_ = frame_type_breakdown(frames_enc)

        print("\n" + "="*45)
        print(f"  I-frames : {ni}  P-frames : {np_}")
        print(f"  GOP={cfg.GOP_SIZE}  QF={cfg.QF}  ratio={ratio:.2f}x")
        print("="*45)

    if args.mode in ('decode', 'all'):
        decode_all(cfg.OUTPUT_BIN)


if __name__ == "__main__":
    main()