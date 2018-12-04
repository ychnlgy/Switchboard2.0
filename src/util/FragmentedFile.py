import numpy, tqdm, os

class FragmentedFile:

    KEY = "!Fragments::"

    def __init__(self, f):
        self.f = f
    
    def load(self):
        with open(self.f, "rb") as f:
            n = self._loadkey(f)
            for i in tqdm.tqdm(range(n), desc="Loading %s" % self.f, ncols=80):
                yield numpy.load(f)
    
    def dump(self, n, it):
        assert not os.path.isfile(self.f)
        with open(self.f, "wb") as f:
            numpy.save(f, FragmentedFile.KEY + str(n))
            it = tqdm.tqdm(it, desc="Dumping to %s" % self.f, ncols=80)
            for i, data in enumerate(it, 1):
                numpy.save(f, data)
        assert i == n
    
    def _loadkey(self, f):
        key = numpy.load(f)
        assert key.startswith(FragmentedFile.KEY)
        num = key.rstrip(FragmentedFile.KEY)
        assert num.isdigit()
        return int(num)
