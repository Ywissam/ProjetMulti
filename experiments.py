import os
import numpy as np
import matplotlib.pyplot as plt

from encoder import load_frames, preprocess_all, encode_all, save_bin
from Globale.entropique import compress_and_save, compute_compression_ratio
import configuration as cfg


def run_with_params(qf, gop_size):
  
    from Globale.DCT         import encoder_iframe, decoder_iframe
    from Globale.ompensation import encoder_pframe, decoder_pframe

    frames_orig = load_frames(cfg.INPUT_FRAMES_DIR)
    frames_pre  = preprocess_all(frames_orig)

    frames_enc = []
    frames_rec = []

    for idx, pf in enumerate(frames_pre):
        if idx % gop_size == 0:
            enc = encoder_iframe(pf, qf=qf)
            rec = decoder_iframe(enc)
        else:
            enc = encoder_pframe(pf, frames_rec[-1],
                                 qf=qf, fenetre=cfg.SEARCH_RANGE)
            rec = decoder_pframe(enc, frames_rec[-1])
        frames_enc.append(enc)
        frames_rec.append(rec)

    # c pour sauvegarder dans un fichier temporaire
    tmp = "Sortie/tmp_exp"
    taille_comp = compress_and_save(frames_enc, tmp)
    ratio, _    = compute_compression_ratio(frames_orig, taille_comp)
    return ratio


def experience_qf():
    qf_list = [10, 20, 30, 50, 70, 90]
    ratios  = []

    print("=== experience : ratio vs QF ===")
    print(f"    GOP fixe a {cfg.GOP_SIZE}")

    for qf in qf_list:
        print(f"  QF={qf} ...", end='', flush=True)
        r = run_with_params(qf=qf, gop_size=cfg.GOP_SIZE)
        ratios.append(r)
        print(f" ratio={r:.2f}x")

    plt.figure(figsize=(8, 5))
    plt.plot(qf_list, ratios, 'o-', color='steelblue', linewidth=2, markersize=9)
    for x, y in zip(qf_list, ratios):
        plt.annotate(f"{y:.1f}x", (x, y),
                     textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=9)
    plt.xlabel("facteur de qualite QF")
    plt.ylabel("ratio de compression")
    plt.title(f"ratio vs QF  (GOP={cfg.GOP_SIZE} fixe)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(qf_list)
    plt.tight_layout()
    plt.savefig("qf_vs_ratio.png", dpi=120)
    print("  graphique sauvegarde : qf_vs_ratio.png")
    plt.show()

    return qf_list, ratios


def experience_gop():
    gop_list = [1, 2, 5, 10, 15, 25]
    ratios   = []

    print("\n=== experience : ratio vs GOP ===")
    print(f"    QF fixe a {cfg.QF}")

    for gop in gop_list:
        print(f"  GOP={gop} ...", end='', flush=True)
        r = run_with_params(qf=cfg.QF, gop_size=gop)
        ratios.append(r)
        print(f" ratio={r:.2f}x")

    plt.figure(figsize=(8, 5))
    plt.plot(gop_list, ratios, 's-', color='darkorange', linewidth=2, markersize=9)
    for x, y in zip(gop_list, ratios):
        plt.annotate(f"{y:.1f}x", (x, y),
                     textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=9)
    plt.xlabel("taille du GOP")
    plt.ylabel("ratio de compression")
    plt.title(f"ratio vs GOP  (QF={cfg.QF} fixe)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(gop_list)
    plt.tight_layout()
    plt.savefig("gop_vs_ratio.png", dpi=120)
    print("  graphique sauvegarde : gop_vs_ratio.png")
    plt.show()

    return gop_list, ratios


def afficher_les_deux(qf_list, qf_ratios, gop_list, gop_ratios):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("partie 5 - analyse experimentale", fontsize=13, fontweight='bold')

    ax1.plot(qf_list, qf_ratios, 'o-', color='steelblue', linewidth=2, markersize=8)
    for x, y in zip(qf_list, qf_ratios):
        ax1.annotate(f"{y:.1f}x", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha='center', fontsize=8)
    ax1.set_xlabel("QF"); ax1.set_ylabel("ratio de compression")
    ax1.set_title(f"ratio vs QF  (GOP={cfg.GOP_SIZE})"); ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xticks(qf_list)

    ax2.plot(gop_list, gop_ratios, 's-', color='darkorange', linewidth=2, markersize=8)
    for x, y in zip(gop_list, gop_ratios):
        ax2.annotate(f"{y:.1f}x", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha='center', fontsize=8)
    ax2.set_xlabel("GOP"); ax2.set_ylabel("ratio de compression")
    ax2.set_title(f"ratio vs GOP  (QF={cfg.QF})"); ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_xticks(gop_list)

    plt.tight_layout()
    plt.savefig("part5_experimental.png", dpi=120, bbox_inches='tight')
    print("graphique combine sauvegarde : part5_experimental.png")
    plt.show()


if __name__ == "__main__":
    qf_list,  qf_ratios  = experience_qf()
    gop_list, gop_ratios = experience_gop()
    afficher_les_deux(qf_list, qf_ratios, gop_list, gop_ratios)

    print("\nexperiences terminees !")
    print("fichiers generes : qf_vs_ratio.png  gop_vs_ratio.png  part5_experimental.png")