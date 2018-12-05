import numpy, math

def warp(slc, oldpts, newpts):
    
    '''
    
    Given a 1D vector, anchor points on this vector
    and corresponding movements of these points,
    elongate and compress areas appropriately.
    
    '''
    
    parts = []
    for p1, p2, q1, q2 in pairmatch(oldpts, newpts):
        parts.append(_warp(slc[p1:p2], q2-q1))
    return numpy.concatenate(parts)

def pairmatch(oldpts, newpts):
    assert len(oldpts) == len(newpts)
    for i in range(len(oldpts)-1):
        yield oldpts[i], oldpts[i+1], newpts[i], newpts[i+1]

# TODO: old and new pts are frequency numbers, not indices

def _warp(slc, length):
    dt = length/len(slc)
    spread = max(math.ceil(dt), 1)
    out = numpy.zeros(length)
    msk = numpy.zeros(length)
    idx = numpy.arange(len(slc))
    newidx = numpy.floor(idx*dt).astype(int)
    msk[newidx] = 1
    out[newidx] = slc[:]
    conv = 2**-numpy.arange(1, spread+1).astype(float)
    conv = numpy.concatenate([numpy.flip(conv), numpy.array([1]), conv])
    print(conv)
    #cout = numpy.convolve(out, conv)/numpy.convolve(msk, conv)
    return out

if __name__ == "__main__":
    
    slc = numpy.arange(20)
    oldpts = [0, 10, 20]
    newpts = [0, 18, 20]
    
    print(warp(slc, oldpts, newpts))
