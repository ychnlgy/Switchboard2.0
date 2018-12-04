import numpy, tqdm

class FragmentedFile:

    KEY = "!Fragments::"

    def __init__(self, f):
        self.f = f
        self.n = None
    
    def __len__(self):
        if self.n is None:
            next(self.load())
        return self.n
    
    def load(self):
        with open(self.f, "rb") as f:
            self.n = self._loadkey(f)
            for i in tqdm.tqdm(range(self.n), desc="Load << %s" % self.f, ncols=80):
                yield numpy.load(f)
    
    def dump(self, n, it):
        self.n = n
        with open(self.f, "wb") as f:
            numpy.save(f, FragmentedFile.KEY + str(n))
            counter = tqdm.tqdm(range(1, n+1), desc="Dump >> %s" % self.f, ncols=80)
            for i, data in zip(counter, it):
                numpy.save(f, data)
        assert i == n
    
    def _loadkey(self, f):
        key = numpy.load(f)
        assert key.startswith(FragmentedFile.KEY)
        num = key.rstrip(FragmentedFile.KEY)
        assert num.isdigit()
        return int(num)
