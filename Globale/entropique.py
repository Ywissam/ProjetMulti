import numpy as np
import os

def rle_encode(data):
 
    resultat = []  
    cur   = data[0]
    count = 1

    for i in range(1, len(data)):
        if data[i] == cur and count < 255:
            count += 1
        else:
            resultat.append((int(cur), count))
            cur   = data[i]
            count = 1

    resultat.append((int(cur), count))
    return resultat


def rle_decode(encoded):
    # on reconstruit le tableau depuis la liste de tuples
    resultat = []
    for valeur, count in encoded:
        resultat.extend([valeur] * count)
    return np.array(resultat, dtype=np.uint8)


def serialiser(frames):
    # convertion de frames
    # on ecrit tout dans une liste puis we concatene
    parts = []

    parts.append(np.array([len(frames)], dtype=np.int32).tobytes())

    for f in frames:  
        parts.append(np.array([0 if f['type']=='I' else 1],
                               dtype=np.uint8).tobytes())
        parts.append(np.array([f['qf']], dtype=np.int32).tobytes())

        for k in ['Y_shape', 'Cb_shape', 'Cr_shape']:
            parts.append(np.array(f[k], dtype=np.int32).tobytes())

        if f['type'] == 'I':
            for k in ['Y_blocs', 'Cb_blocs', 'Cr_blocs']:
                arr = f[k]
                parts.append(np.array(arr.shape, dtype=np.int32).tobytes())
                parts.append(arr.astype(np.int16).tobytes())
        else:
            mv = f['mvs']
            parts.append(np.array(mv.shape, dtype=np.int32).tobytes())
            parts.append(mv.astype(np.int8).tobytes())

            for k in ['Y_res_blocs', 'Cb_res_blocs', 'Cr_res_blocs']:
                arr = f[k]
                parts.append(np.array(arr.shape, dtype=np.int32).tobytes())
                parts.append(arr.astype(np.int16).tobytes())

    raw = b''.join(parts)
    return np.frombuffer(raw, dtype=np.uint8)


def deserialiser(data):
    raw = data.tobytes()
    pos = 0

    def lire(n, dtype):
        nonlocal pos
        val = np.frombuffer(raw[pos:pos+n], dtype=dtype)
        pos += n
        return val

    n_frames = int(lire(4, np.int32)[0])
    frames   = []

    for _ in range(n_frames):
        f = {}
        t = int(lire(1, np.uint8)[0])
        f['type'] = 'I' if t == 0 else 'P'
        f['qf']   = int(lire(4, np.int32)[0])

        for k in ['Y_shape', 'Cb_shape', 'Cr_shape']:
            f[k] = tuple(lire(8, np.int32))

        if f['type'] == 'I':
            for k in ['Y_blocs', 'Cb_blocs', 'Cr_blocs']:
                sh = tuple(lire(16, np.int32))
                nb = int(np.prod(sh))
                f[k] = lire(nb*2, np.int16).reshape(sh)
        else:
            sh_mv = tuple(lire(12, np.int32))
            nb_mv = int(np.prod(sh_mv))
            f['mvs'] = lire(nb_mv, np.int8).reshape(sh_mv)
            for k in ['Y_res_blocs', 'Cb_res_blocs', 'Cr_res_blocs']:
                sh = tuple(lire(16, np.int32))
                nb = int(np.prod(sh))
                f[k] = lire(nb*2, np.int16).reshape(sh)

        frames.append(f)

    return frames


def compress_and_save(frames, chemin):
    print("\n--- partie 4 : compression RLE ---")

    raw = serialiser(frames)
    print(f"  taille avant RLE : {len(raw)/1024:.1f} Ko")

    encoded = rle_encode(raw)

    taille_comp = len(encoded) * 2
    taux = (1 - taille_comp/len(raw)) * 100
    print(f"  taille apres RLE : {taille_comp/1024:.1f} Ko")
    print(f"  taux de compression : {taux:.1f}%")

    vals   = np.array([v for v, c in encoded], dtype=np.uint8)
    counts = np.array([c for v, c in encoded], dtype=np.uint8)

    os.makedirs(os.path.dirname(chemin) or '.', exist_ok=True)
    np.savez(chemin, vals=vals, counts=counts)
    print(f"  fichier : {chemin}.npz")

    return taille_comp


def load_and_decompress(chemin):
    path = chemin if chemin.endswith('.npz') else chemin + '.npz'
    data = np.load(path, allow_pickle=False)

    encoded = list(zip(data['vals'].tolist(), data['counts'].tolist()))

    raw    = rle_decode(encoded)
    frames = deserialiser(raw)
    print(f"  {len(frames)} frames chargees depuis {path}")
    return frames


def compute_compression_ratio(frames_bgr, taille_comp):
    taille_orig = sum(f.nbytes for f in frames_bgr)
    ratio = taille_orig / taille_comp
    return ratio, taille_orig