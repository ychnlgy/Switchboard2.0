import numpy

def onehot(indices, size):
    N = len(indices)
    out = numpy.zeros((N, size))
    out[numpy.arange(N), indices] = 1.0
    return out

def unittest():
    i = numpy.array([1, 2, 4])
    a = onehot(i, 5)
    assert (a == numpy.array([
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1]
    ])).all()
    
    i = numpy.array([
        [1, 2, 4],
        [0, 1, 2]
    ])
    a = onehot(i, 5)
    assert (a == numpy.array([
        [
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1]
        ],
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0]
        ]
    ])).all()
