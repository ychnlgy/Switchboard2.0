#!/usr/bin/python3

import numpy, tqdm, json, os, random
import scipy.signal

import util

import speech_features

RATE = 8000
NUMCEP = 16
CLASSES = 45

LENGTH = 4000
DELTA = 1000
LABEL_SCALE = 100

LABEL_SAVE_JSON = "switchboard-labels.json"
OUT_FILE = "melspecs-switchboard.npy"
EMPTY = "<EMPTY>"
SIL = "SIL"
SIL_DROPOUT = 0.5

def load(specf):
    specf = os.path.join(specf, OUT_FILE)
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
    
    kmap, imap = load_label_map(os.path.join(specf, LABEL_SAVE_JSON))
    size = len(kmap)
    
    for i, (x, y, l) in zip(range(SAMPLES), data):
        axes[0, i].imshow(x.T, cmap="hot", interpolation="bicubic", aspect="auto")
        y = util.onehot(y, size).T
        for j in range(len(y)):
            axes[1, i].plot(y[j])
        axes[0, i].set_title(str(l.tolist()))
    
    pyplot.savefig("switchboard-mfcc-samples.png", bbox_inches="tight")

def create_spectrograms(dataf):
    data = list(_load(dataf))
    
    Xa = []
    Xb = []
    ya = []
    yb = []
    la = []
    lb = []
    all_labels = set()
    
    for num, rate, waveA, waveB, pA, pB, sA, sB in tqdm.tqdm(data, desc="Processing data", ncols=80):
        assert rate == RATE
        waveA = remove_noise(waveA)
        waveB = remove_noise(waveB)
        yA = match_labels(waveA, pA)
        yB = match_labels(waveB, pB)
        
        for wavA, slcA, slcy in slice_step(waveA, yA, pA, LENGTH, DELTA):
            if keep_slice(slcA):
                melA = convert_spectrogram(wavA)
                Xa.append(melA)
                ya.append(slcA)
                la.append(slcy)
                all_labels.update(slcA)
        
        for wavB, slcB, slcy in slice_step(waveB, yB, pB, LENGTH, DELTA):
            if keep_slice(slcB):
                melB = convert_spectrogram(wavB)
                Xb.append(melB)
                yb.append(slcB)
                lb.append(slcy)
                all_labels.update(slcB)
    
    print('''
    ***
    
    Skipped %d files because they were shorter than 1 second.
    
    ***
    ''' % SKIPPED)
    
    all_labels = sorted(all_labels)
    assert all_labels[0] == EMPTY
    assert len(all_labels) == CLASSES + 1
    
    DATA_DIR = os.path.dirname(dataf)
    
    label_file = os.path.join(DATA_DIR, LABEL_SAVE_JSON)
    save_label_map(all_labels, label_file)
    keymap, idxmap = load_label_map(label_file)
    ya = convert_key2idx(keymap, ya)
    yb = convert_key2idx(keymap, yb)
    la = convert_key2idx(keymap, la)
    lb = convert_key2idx(keymap, lb)
    assert len(Xa) == len(ya) == len(la)
    assert len(Xb) == len(yb) == len(lb)
    
    out_file = os.path.join(DATA_DIR, OUT_FILE)
    X = Xa + Xb
    Y = ya + yb
    L = la + lb
    assert len(X) == len(Y) == len(L)
    L = pad_fitlargest(L)
    
    fragfile = util.FragmentedFile(out_file)
    fragfile.dump(len(X), zip(X, Y, L))

# === PRIVATE ===

SKIPPED = 0

def pad_fitlargest(labels):
    "Because the first class is EMPTY, we can use that as padding."
    longest = max(map(len, labels))
    print("Labels padded to length: %d" % longest)
    def pad(arr):
        out = numpy.zeros(longest).astype(numpy.int32)
        out[:len(arr)] = arr
        return out
    return list(map(pad, labels))

def keep_slice(slc):
    if all([v==SIL for v in slc]):
        return random.random() > SIL_DROPOUT
    else:
        return True

def slice_step(wav, lab, phns, length, step):
    if len(wav) == len(lab) and len(wav) > length:
    
        def locate_phns(i):
            out = []
            for name, start, end, pid in phns:
                if start > end:
                    start, end = end, start
                if start >= i+step:
                    break
                elif end < i:
                    continue
                else:
                    out.append(name)
            return out
    
        d, r = divmod(len(wav)-length, step)
        for i in range(0, d*step, step):
            yield wav[i:i+length], lab[i:i+length][::LABEL_SCALE], locate_phns(i)
        if r:
            yield wav[-length:], lab[-length:][::LABEL_SCALE], locate_phns(len(wav)-length)
    else:
        global SKIPPED
        SKIPPED += 1

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
        labels = json.load(f)
        print("Classes: %d" % (len(labels)-1))
        assert CLASSES + 1 == len(labels)
        idxmap = dict(enumerate(labels))
        keymap = {k:i for i, k in idxmap.items()}
        return keymap, idxmap

def _load(dataf):
    with open(dataf, "rb") as f:
        i = 5
        with tqdm.tqdm(desc="Loading %s" % dataf, ncols=80) as bar:
            while True:
                try:
                    yield numpy.load(f)
                    i -= 1
                    if i <= 0: break
                    bar.update()
                except OSError:
                    break

def convert_spectrogram(wav):
    return speech_features.mfcc(wav, samplerate=RATE, numcep=NUMCEP)

def match_labels(wav, phns):
    y = [EMPTY] * len(wav)
    for name, start, end, pid in phns:
        if start > end:
            start, end = end, start
        y[start:end] = [name] * (end-start)
    return y

@util.main(__name__)
def main(fname, sample=0):
    sample = int(sample)
    if sample:
        view(fname)
    else:
        create_spectrograms(fname)
