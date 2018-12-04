import numpy, tqdm, json

import speech_features

RATE = 8000
NUMCEP = 16

LABEL_SAVE_JSON = "switchboard-labels.json"

def load(dataf):
    "Returns paired X and y for speakers A and B."
    data = list(_load(dataf))
    
    Xa = []
    Xb = []
    ya = []
    yb = []
    all_labels = set()
    
    for num, rate, waveA, waveB, pA, pB, sA, sB in tqdm.tqdm(data, desc="Processing data", ncols=80):
        assert rate == RATE
        melA = convert_spectrogram(waveA)
        melB = convert_spectrogram(waveB)
        labA = match_labels(waveA, melA, pA)
        labB = match_labels(waveB, melB, pB)
        
        Xa.append(melA)
        Xb.append(melB)
        ya.append(labA)
        yb.append(labB)
    
        all_labels.update(labA + labB)
    
    all_labels = sorted(all_labels)
    save_label_map(all_labels, LABEL_SAVE_JSON)
    keymap, idxmap = load_label_map(LABEL_SAVE_JSON)
    ya = convert_key2idx(keymap, ya)
    yb = convert_key2idx(keymap, yb)
    return (
        numpy.array(Xa),
        numpy.array(Xb),
        numpy.array(ya),
        numpy.array(yb)
    )

# === PRIVATE ===

def convert_key2idx(keymap, y):
    out = []
    for arr in y:
        out.append([keymap[v] for v in arr])
    return out

def save_label_map(labels, fname):
    with open(fname, "w") as f:
        json.dump(labels, f)

def load_label_map(fname):
    with open(fname, "r") as f:
        labels = json.load(f)
        idxmap = dict(enumerate(labels))
        keymap = {k:i for i, k in idxmap.items()}
        return keymap, idxmap

def _load(dataf):
    with open(dataf, "rb") as f:
        with tqdm.tqdm(desc="Loading %s" % dataf, ncols=80) as bar:
            while True:
                try:
                    yield numpy.load(f)
                    bar.update()
                except OSError:
                    break

def convert_spectrogram(wav):
    return speech_features.mfcc(wav, samplerate=RATE, numcep=NUMCEP)

def match_labels(wav, mel, phns):
    t0 = wav.shape[0]
    tf = mel.shape[0]
    ratio = tf/t0
    labels = []
    for name, start, end, pid in phns:
        ci = int(round(start*ratio))
        cf = int(round(end*ratio))
        if not cf > ci:
            print(cf, ci, start, end, ratio)
            input()
            assert cf == ci
            continue
        labels.extend((cf-ci)*[name])
    return labels
