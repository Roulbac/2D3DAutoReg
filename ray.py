import numpy as np

class Ray(object):
    def __init__(self, src, dst):
        self.src, self.dst = np.array(src), np.array(dst)

    def get_dir(self):
        return (self.dst-self.src)/np.norm(self.dest-self.src)
