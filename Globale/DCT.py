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


def get_quantization_matrix(channel='luma', qf=50):
    base = Q_LUMA if channel == 'luma' else Q_CHROMA

    if qf < 50:
        scale = 5000 / qf
    else:
        scale = 200 - 2 * qf

    Q = np.floor((base * scale + 50) / 100)
    Q = np.clip(Q, 1, 255)

    return Q.astype(np.float32)


def dct2d(block):
    return cv2.dct(block.astype(np.float32))


def idct2d(block):
    return cv2.idct(block.astype(np.float32))


def encode_channel(channel, Q):
    H, W = channel.shape

    # Padding pour multiple de 8
    H_pad = (8 - H % 8) % 8
    W_pad = (8 - W % 8) % 8

    padded = np.pad(channel.astype(np.float32) - 128,
                    ((0, H_pad), (0, W_pad)),
                    mode='edge')

    Hp, Wp = padded.shape
    bh, bw = Hp // 8, Wp // 8

    quantized = np.zeros((bh, bw, 8, 8), dtype=np.int16)

    for i in range(bh):
        for j in range(bw):
            block = padded[i*8:(i+1)*8, j*8:(j+1)*8]

            dct_block = dct2d(block)
            quantized[i, j] = np.round(dct_block / Q)

    return quantized


def decode_channel(quantized_blocks, Q, original_shape):
    bh, bw = quantized_blocks.shape[:2]

    recon = np.zeros((bh * 8, bw * 8), dtype=np.float32)

    for i in range(bh):
        for j in range(bw):
            dct_block = quantized_blocks[i, j] * Q

            block = idct2d(dct_block)
            recon[i*8:(i+1)*8, j*8:(j+1)*8] = block

    H, W = original_shape
    recon = recon[:H, :W] + 128

    return np.clip(recon, 0, 255).astype(np.uint8)


def encode_iframe(processed_frame, qf=50):
    Qy = get_quantization_matrix('luma', qf)
    Qc = get_quantization_matrix('chroma', qf)

    return {
        'type': 'I',
        'Y': encode_channel(processed_frame['Y'], Qy),
        'Cb': encode_channel(processed_frame['Cb'], Qc),
        'Cr': encode_channel(processed_frame['Cr'], Qc),
        'Y_shape': processed_frame['Y'].shape,
        'Cb_shape': processed_frame['Cb'].shape,
        'Cr_shape': processed_frame['Cr'].shape,
        'qf': qf
    }


def decode_iframe(encoded):
    qf = encoded['qf']

    Qy = get_quantization_matrix('luma', qf)
    Qc = get_quantization_matrix('chroma', qf)

    return {
        'Y': decode_channel(encoded['Y'], Qy, encoded['Y_shape']),
        'Cb': decode_channel(encoded['Cb'], Qc, encoded['Cb_shape']),
        'Cr': decode_channel(encoded['Cr'], Qc, encoded['Cr_shape'])
    }