import numpy

class Vector2D:
    def __init__(self, nparr):
        self.nparr = nparr

    @staticmethod
    def coords_to_nparr(x, y):
        return numpy.array([x, y])

    def __array__(self):
        return self.nparr

    def get_x(self):
        return self.nparr[0, 0]

    def get_y(self):
        return self.nparr[1, 1]


class Vector3D:

    def __init__(self, nparr):
        self.nparr = nparr

    @staticmethod
    def coords_to_nparr(x, y, z):
        return numpy.array([x, y, z])

    def __array__(self):
        return self.nparr

    # FIXME this is terribly inefficient
    @staticmethod
    def plus_i():
        return Vector3D(numpy.array([1, 0, 0]))

    @staticmethod
    def plus_j():
        return Vector3D(numpy.array([0, 1, 0]))

    @staticmethod
    def plus_k():
        return Vector3D(numpy.array([0, 0, 1]))