import numpy as np
import cv2
import matplotlib.pyplot as plt
from Globale.DCT import get_Q, dct_bloc, idct_bloc
from Globale.ompensation import TAILLE_MB


def psnr(orig, recon):
    mse = np.mean((orig.astype(np.float64) - recon.astype(np.float64))**2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(255.0**2 / mse)


def visualize_pipeline(frames_orig, frames_pre, frames_enc, frames_rec):
    from Globale.convertionCouleur import bgr_to_ycbcr, postprocess_frame

    fig = plt.figure(figsize=(24, 18))
    fig.suptitle("Pipeline MPEG-4 - Visualisation complete",
                 fontsize=13, fontweight='bold')

    for k in range(4):
        ax = fig.add_subplot(4, 4, k+1)
        if k < len(frames_orig):
            ax.imshow(cv2.cvtColor(frames_orig[k], cv2.COLOR_BGR2RGB))
            ax.set_title(f"frame {k+1}", fontsize=9)
        ax.axis('off')

    Y, Cb, Cr = bgr_to_ycbcr(frames_orig[0])
    items_l2 = [
        (cv2.cvtColor(frames_orig[0], cv2.COLOR_BGR2RGB), "original RGB",  None),
        (Y,  "canal Y",  'gray'),
        (Cb, "canal Cb", 'Blues_r'),
        (Cr, "canal Cr", 'Reds_r'),
    ]
    for k, (img, titre, cmap) in enumerate(items_l2):
        ax = fig.add_subplot(4, 4, 4+k+1)
        ax.imshow(img, cmap=cmap)
        ax.set_title(titre, fontsize=9)
        ax.axis('off')

    qf  = frames_enc[0]['qf']
    Q   = get_Q('luma', qf)
    bi, bj = 5, 5
    bloc = frames_pre[0]['Y'].astype(np.float32)[bi*8:(bi+1)*8, bj*8:(bj+1)*8]

    dct_b   = dct_bloc(bloc - 128)
    quant_b = np.int16(dct_b / Q)
    recon_b = np.clip(idct_bloc(quant_b.astype(np.float32) * Q) + 128, 0, 255)

    dct_show = np.log(1 + np.abs(dct_b))

    items_l3 = [
        (bloc,              f"pixels bruts\nbloc [{bi},{bj}]", 'gray'),
        (dct_show,          "DCT (echelle log)",               'hot'),
        (np.abs(quant_b),   f"apres quantification\nQF={qf}", 'hot'),
        (recon_b,           "bloc reconstruit",                 'gray'),
    ]
    for k, (img, titre, cmap) in enumerate(items_l3):
        ax = fig.add_subplot(4, 4, 8+k+1)
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(titre, fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    p_idx = next((i for i, e in enumerate(frames_enc) if e['type']=='P'), None)

    ax_mv = fig.add_subplot(4, 4, 13)
    if p_idx is not None:
        enc_p   = frames_enc[p_idx]
        rec_p   = frames_rec[p_idx]
        rec_ref = frames_rec[p_idx-1]
        ref_bgr = postprocess_frame(rec_ref['Y'], rec_ref['Cb'], rec_ref['Cr'])
        ax_mv.imshow(cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB))
        mvs  = enc_p['mvs']
        H, W = ref_bgr.shape[:2]
        MB   = TAILLE_MB
        for i in range(mvs.shape[0]):
            for j in range(mvs.shape[1]):
                cx, cy = j*MB+MB//2, i*MB+MB//2
                dy, dx  = int(mvs[i,j,0]), int(mvs[i,j,1])
                if dy != 0 or dx != 0:
                    ax_mv.annotate("", xy=(cx+dx, cy+dy), xytext=(cx,cy),
                                   arrowprops=dict(arrowstyle='->',
                                                   color='cyan', lw=0.8))
        ax_mv.set_xlim(0,W); ax_mv.set_ylim(H,0)
        ax_mv.set_title(f"vecteurs mouvement\npframe {p_idx+1}", fontsize=9)
    ax_mv.axis('off')

    ax_res = fig.add_subplot(4, 4, 14)
    if p_idx is not None:
        res = rec_p['Y'].astype(np.float32) - rec_ref['Y'].astype(np.float32)
        im_r = ax_res.imshow(res, cmap='RdBu', vmin=-40, vmax=40)
        ax_res.set_title(f"residual Y\npframe {p_idx+1}", fontsize=9)
        plt.colorbar(im_r, ax=ax_res, fraction=0.046, pad=0.04)
    ax_res.axis('off')

    ax_rec = fig.add_subplot(4, 4, 15)
    if p_idx is not None:
        recon_bgr = postprocess_frame(rec_p['Y'], rec_p['Cb'], rec_p['Cr'])
        score     = psnr(frames_orig[p_idx], recon_bgr)
        ax_rec.imshow(cv2.cvtColor(recon_bgr, cv2.COLOR_BGR2RGB))
        ax_rec.set_title(f"frame {p_idx+1} reconstruite\nPSNR={score:.1f}dB",
                         fontsize=9)
    ax_rec.axis('off')

    ax_bar = fig.add_subplot(4, 4, 16)
    types = [e['type'] for e in frames_enc]
    ni  = types.count('I')
    np_ = types.count('P')
    ax_bar.bar(['I-frames','P-frames'], [ni, np_],
               color=['steelblue','tomato'], width=0.5)
    ax_bar.set_title(f"repartition\n{ni} I  +  {np_} P", fontsize=9)
    for sp in ['top','right']:
        ax_bar.spines[sp].set_visible(False)

    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig("pipeline_visualisation.png", dpi=120, bbox_inches='tight')
    print("figure sauvegardee : pipeline_visualisation.png")
    plt.show()


def frame_type_breakdown(frames_enc):
    ni  = sum(1 for e in frames_enc if e['type']=='I')
    np_ = sum(1 for e in frames_enc if e['type']=='P')
    return ni, np_