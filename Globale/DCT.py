import numpy as np
import cv2

Q_LUMA = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103,  77],
    [24, 35, 55, 64, 81, 104, 113,  92],
    [49, 64, 78, 87,103, 121, 120, 101],
    [72, 92, 95, 98,112, 100, 103,  99],
], dtype=np.float32)

Q_CHROMA = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float32)


def get_Q(canal='luma', qf=50):
 
    base = Q_LUMA if canal == 'luma' else Q_CHROMA
    if qf < 50:
        scale = 5000 / qf
    else:
        scale = 200 - 2*qf
    Q = np.floor((base * scale + 50) / 100)
    return np.clip(Q, 1, 255).astype(np.float32)


def dct_bloc(bloc):

    return cv2.dct(np.float32(bloc))


def idct_bloc(bloc):
    return cv2.idct(np.float32(bloc))


def encoder_canal(canal, Q):

    H, W = canal.shape

    haut  = (8 - H % 8) % 8
    droite = (8 - W % 8) % 8
    img_pad = np.pad(canal.astype(np.float32) - 128,
                     ((0, haut), (0, droite)), 'constant')

    bh = img_pad.shape[0] // 8
    bw = img_pad.shape[1] // 8

    blocs = np.zeros((bh, bw, 8, 8), dtype=np.int16)

    for i in range(bh):
        for j in range(bw):
            bloc = img_pad[i*8:(i+1)*8, j*8:(j+1)*8]
            dct  = dct_bloc(bloc)
            blocs[i, j] = np.int16(dct / Q)

    return blocs, (H, W)


def decoder_canal(blocs, Q, shape):

    bh, bw = blocs.shape[:2]
    img = np.zeros((bh*8, bw*8), dtype=np.float32)

    for i in range(bh):
        for j in range(bw):
            dct  = blocs[i, j].astype(np.float32) * Q
            bloc = idct_bloc(dct)
            img[i*8:(i+1)*8, j*8:(j+1)*8] = bloc

    H, W = shape
    return np.clip(img[:H, :W] + 128, 0, 255).astype(np.uint8)


def encoder_iframe(frame, qf=50):
    Qy = get_Q('luma',   qf)
    Qc = get_Q('chroma', qf)

    Y_blocs,  Y_shape  = encoder_canal(frame['Y'],  Qy)
    Cb_blocs, Cb_shape = encoder_canal(frame['Cb'], Qc)
    Cr_blocs, Cr_shape = encoder_canal(frame['Cr'], Qc)

    return {
        'type': 'I', 'qf': qf,
        'Y_blocs':  Y_blocs,  'Y_shape':  Y_shape,
        'Cb_blocs': Cb_blocs, 'Cb_shape': Cb_shape,
        'Cr_blocs': Cr_blocs, 'Cr_shape': Cr_shape,
    }


def decoder_iframe(enc):
    qf = enc['qf']
    Qy = get_Q('luma',   qf)
    Qc = get_Q('chroma', qf)

    Y  = decoder_canal(enc['Y_blocs'],  Qy, enc['Y_shape'])
    Cb = decoder_canal(enc['Cb_blocs'], Qc, enc['Cb_shape'])
    Cr = decoder_canal(enc['Cr_blocs'], Qc, enc['Cr_shape'])

    return {'Y': Y, 'Cb': Cb, 'Cr': Cr}