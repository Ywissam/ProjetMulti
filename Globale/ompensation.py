import numpy as np
from Globale.DCT import get_Q, dct_bloc, idct_bloc

# partie 3 - P-frames
# estimation de mouvement avec block matching
TAILLE_MB = 16


def block_matching(bloc_cur, ref_Y, y0, x0, fenetre):
    # we search le bloc le plus similaire dans ref_Y
    H, W = ref_Y.shape
    bh, bw = bloc_cur.shape
    best_mv  = (0, 0)
    best_sad = float('inf')

    for dy in range(-fenetre, fenetre+1):
        for dx in range(-fenetre, fenetre+1):
            ry, rx = y0+dy, x0+dx
            if ry < 0 or rx < 0 or ry+bh > H or rx+bw > W:
                continue
            bloc_ref = ref_Y[ry:ry+bh, rx:rx+bw]
            sad = int(np.sum(np.abs(
                bloc_cur.astype(np.int16) - bloc_ref.astype(np.int16)
            )))
            if sad < best_sad:
                best_sad = sad
                best_mv  = (dy, dx)

    return best_mv


def encoder_residual(res, Q):
   
    H, W = res.shape
    haut   = (8 - H % 8) % 8
    droite = (8 - W % 8) % 8
    img_pad = np.pad(res.astype(np.float32),
                     ((0, haut), (0, droite)), 'constant')
    bh = img_pad.shape[0] // 8
    bw = img_pad.shape[1] // 8
    blocs = np.zeros((bh, bw, 8, 8), dtype=np.int16)
    for i in range(bh):
        for j in range(bw):
            dct = dct_bloc(img_pad[i*8:(i+1)*8, j*8:(j+1)*8])
            blocs[i, j] = np.int16(dct / Q)
    return blocs, (H, W)


def decoder_residual(blocs, Q, shape):
    bh, bw = blocs.shape[:2]
    img = np.zeros((bh*8, bw*8), dtype=np.float32)
    for i in range(bh):
        for j in range(bw):
            img[i*8:(i+1)*8, j*8:(j+1)*8] = idct_bloc(
                blocs[i, j].astype(np.float32) * Q
            )
    H, W = shape
    return img[:H, :W]


def predire_chroma(chroma_ref, mvs, shape):
    # on applique les vecteurs de mouvement sur les canaux couleur
    # divise par 2 car 4:2:0
    H, W = shape
    MB_c = TAILLE_MB // 2
    pred = np.zeros((H, W), dtype=np.int16)
    nb_h, nb_w = mvs.shape[:2]
    for i in range(nb_h):
        for j in range(nb_w):
            y0, x0 = i*MB_c, j*MB_c
            y1, x1 = min(y0+MB_c, H), min(x0+MB_c, W)
            dy = int(mvs[i, j, 0]) // 2
            dx = int(mvs[i, j, 1]) // 2
            ry = max(0, min(y0+dy, chroma_ref.shape[0]-(y1-y0)))
            rx = max(0, min(x0+dx, chroma_ref.shape[1]-(x1-x0)))
            pred[y0:y1, x0:x1] = chroma_ref[ry:ry+(y1-y0), rx:rx+(x1-x0)]
    return pred


def encoder_pframe(frame, ref, qf=50, fenetre=8):
    Qy = get_Q('luma',   qf)
    Qc = get_Q('chroma', qf)

    Y_cur  = frame['Y'].astype(np.int16)
    Cb_cur = frame['Cb'].astype(np.int16)
    Cr_cur = frame['Cr'].astype(np.int16)
    Y_ref  = ref['Y']
    Cb_ref = ref['Cb'].astype(np.int16)
    Cr_ref = ref['Cr'].astype(np.int16)

    H, W = Y_cur.shape
    MB   = TAILLE_MB
    nb_h = (H + MB-1) // MB
    nb_w = (W + MB-1) // MB


    mvs       = np.zeros((nb_h, nb_w, 2), dtype=np.int8)
    residual_Y = np.zeros((H, W), dtype=np.float32)

    for i in range(nb_h):
        for j in range(nb_w):
            y0, x0 = i*MB, j*MB
            y1, x1 = min(y0+MB, H), min(x0+MB, W)

            bloc_cur = Y_cur[y0:y1, x0:x1].astype(np.uint8)
            dy, dx   = block_matching(bloc_cur, Y_ref, y0, x0, fenetre)
            mvs[i, j] = [dy, dx]

            # residual = frame actuelle - prediction
            ry = max(0, min(y0+dy, H-(y1-y0)))
            rx = max(0, min(x0+dx, W-(x1-x0)))
            pred = Y_ref[ry:ry+(y1-y0), rx:rx+(x1-x0)].astype(np.int16)
            residual_Y[y0:y1, x0:x1] = Y_cur[y0:y1, x0:x1] - pred

    Cb_pred = predire_chroma(Cb_ref, mvs, Cb_cur.shape)
    Cr_pred = predire_chroma(Cr_ref, mvs, Cr_cur.shape)
    Cb_res  = (Cb_cur - Cb_pred).astype(np.float32)
    Cr_res  = (Cr_cur - Cr_pred).astype(np.float32)

    Y_res_blocs,  _ = encoder_residual(residual_Y, Qy)
    Cb_res_blocs, _ = encoder_residual(Cb_res,     Qc)
    Cr_res_blocs, _ = encoder_residual(Cr_res,     Qc)

    return {
        'type': 'P', 'qf': qf,
        'mvs':          mvs,
        'Y_res_blocs':  Y_res_blocs,
        'Cb_res_blocs': Cb_res_blocs,
        'Cr_res_blocs': Cr_res_blocs,
        'Y_shape':  frame['Y'].shape,
        'Cb_shape': frame['Cb'].shape,
        'Cr_shape': frame['Cr'].shape,
    }


def decoder_pframe(enc, ref):
    qf = enc['qf']
    Qy = get_Q('luma',   qf)
    Qc = get_Q('chroma', qf)

    mvs     = enc['mvs']
    Y_shape  = enc['Y_shape']
    Cb_shape = enc['Cb_shape']
    Cr_shape = enc['Cr_shape']
    H, W = Y_shape
    MB   = TAILLE_MB

    Y_ref  = ref['Y']
    Cb_ref = ref['Cb'].astype(np.int16)
    Cr_ref = ref['Cr'].astype(np.int16)

    Y_pred = np.zeros(Y_shape, dtype=np.int16)
    nb_h, nb_w = mvs.shape[:2]
    for i in range(nb_h):
        for j in range(nb_w):
            y0, x0 = i*MB, j*MB
            y1, x1 = min(y0+MB, H), min(x0+MB, W)
            dy, dx  = int(mvs[i,j,0]), int(mvs[i,j,1])
            ry = max(0, min(y0+dy, H-(y1-y0)))
            rx = max(0, min(x0+dx, W-(x1-x0)))
            Y_pred[y0:y1, x0:x1] = Y_ref[ry:ry+(y1-y0), rx:rx+(x1-x0)]

    Cb_pred = predire_chroma(Cb_ref, mvs, Cb_shape)
    Cr_pred = predire_chroma(Cr_ref, mvs, Cr_shape)

    Y_res  = decoder_residual(enc['Y_res_blocs'],  Qy, Y_shape)
    Cb_res = decoder_residual(enc['Cb_res_blocs'], Qc, Cb_shape)
    Cr_res = decoder_residual(enc['Cr_res_blocs'], Qc, Cr_shape)

    Y  = np.clip(Y_pred  + Y_res,  0, 255).astype(np.uint8)
    Cb = np.clip(Cb_pred + Cb_res, 0, 255).astype(np.uint8)
    Cr = np.clip(Cr_pred + Cr_res, 0, 255).astype(np.uint8)

    return {'Y': Y, 'Cb': Cb, 'Cr': Cr}