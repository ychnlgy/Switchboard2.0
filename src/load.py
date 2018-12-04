#!/usr/bin/python3

import numpy, tqdm, json, os
import scipy.signal

import util

import speech_features

RATE = 8000
NUMCEP = 16

LENGTH = 8000
STEP = 2000

LABEL_SAVE_JSON = "switchboard-labels.json"
OUT_FILE = "melspecs-switchboard.npy"
EMPTY = "<EMPTY>"

def load(specf):
    fragfile = util.FragmentedFile(specf)
    return fragfile.load()

def view(specf):
    SAMPLES = 5
    
    import matplotlib
    matplotlib.use("agg")
    from matplotlib import pyplot
    
    data = load(specf)
    
    fig, axes = pyplot.subplots(nrows=2, ncols=SAMPLES)
    fig.set_size_inches(18, 6)
    
    kmap, imap = load_label_map(os.path.join(os.path.dirname(specf), LABEL_SAVE_JSON))
    size = len(kmap)
    
    for i, (x, y) in zip(range(SAMPLES), data):
        axes[0, i].imshow(x.T, cmap="hot", interpolation="bicubic", aspect="auto")
        y = util.onehot(y, size).T
        for j in range(len(y)):
            axes[1, i].plot(y[j])
    
    pyplot.savefig("switchboard-mfcc-samples.png", bbox_inches="tight")

def create_spectrograms(dataf):
    data = list(_load(dataf))
    
    Xa = []
    Xb = []
    ya = []
    yb = []
    all_labels = set()
    
    for num, rate, waveA, waveB, pA, pB, sA, sB in tqdm.tqdm(data, desc="Processing data", ncols=80):
        assert rate == RATE
        waveA = remove_noise(waveA)
        waveB = remove_noise(waveB)
        labA = match_labels(waveA, pA)
        labB = match_labels(waveB, pB)
        
        for wavA, slcA in slice_step(waveA, labA, LENGTH, STEP, "A"):
            melA = convert_spectrogram(wavA)
            Xa.append(melA)
            ya.append(slcA)
        
        for wavB, slcB in slice_step(waveB, labB, LENGTH, STEP, "B"):
            melB = convert_spectrogram(wavB)
            Xb.append(melB)
            yb.append(slcB)
    
        all_labels.update(labA + labB)
    
    all_labels = sorted(all_labels)
    
    DATA_DIR = os.path.dirname(dataf)
    
    label_file = os.path.join(DATA_DIR, LABEL_SAVE_JSON)
    save_label_map(all_labels, label_file)
    keymap, idxmap = load_label_map(label_file)
    ya = convert_key2idx(keymap, ya)
    yb = convert_key2idx(keymap, yb)
    assert len(Xa) == len(Xb) == len(ya) == len(yb)
    
    out_file = os.path.join(DATA_DIR, OUT_FILE)
    X = Xa + Xb
    Y = ya + yb
    assert len(X) == len(Y)
    
    fragfile = util.FragmentedFile(out_file)
    fragfile.dump(len(X), zip(X, Y))

# === PRIVATE ===

def slice_step(wav, lab, length, step, name):
    assert len(wav) == len(lab) > length
    d, r = divmod(len(wav)-length, step)
    for i in tqdm.tqdm(range(0, d*step, step), desc="Slicing %s waves" % name, ncols=80):
        yield wav[i:i+length], lab[i:i+length]
    if r:
        yield wav[-length:], lab[-length:]

def remove_noise(data):
    b, a = scipy.signal.butter(2, 40/(8000/2), btype="highpass")
    data = scipy.signal.lfilter(b, a, data)
    return data

def convert_key2idx(keymap, y):
    out = []
    for arr in tqdm.tqdm(y, desc="Converting labels to ints", ncols=80):
        out.append(numpy.array([keymap[v] for v in arr]))
    return out

def save_label_map(labels, fname):
    with open(fname, "w") as f:
        json.dump(labels, f)

def load_label_map(fname):
    with open(fname, "r") as f:
        labels = json.load(f) + [EMPTY]
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

def match_labels(wav, phns):
    labels = [EMPTY] * len(wav)
    for name, start, end, pid in phns:
        if start > end:
            start, end = end, start
        labels[start:end] = [name] * (end-start)
    return labels

@util.main(__name__)
def main(fname, sample=0):
    sample = int(sample)
    if sample:
        view(fname)
    else:
        create_spectrograms(fname)
