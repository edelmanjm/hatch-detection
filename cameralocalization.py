import math
import numpy
from datastructures import Vector2D, Vector3D
from scipy.spatial.transform import Rotation
from sympy.geometry import Line3D


def angles_from_screen_space(normalized_screen_point: Vector2D, hoz_fov, vert_fov):
    vpw = 2 * math.tan(hoz_fov / 2)
    vph = 2 * math.tan(vert_fov / 2)

    screen_space_x = vpw / 2 * -normalized_screen_point.get_x()
    screen_space_y = vph / 2 * normalized_screen_point.get_y()

    return Vector2D(numpy.array([
        math.pi / 2 - math.atan2(1, screen_space_x),
        math.pi / 2 - math.atan2(1, screen_space_y)
    ]))


def line(screen_angles: Vector2D, camera_position: Vector3D, camera_rotaiton: Rotation):
    yaw = screen_angles.get_x()
    pitch = screen_angles.get_y()

    pitch_rot = Rotation.from_euler('y', pitch)
    yaw_rot = Rotation.from_euler('z', yaw)

    pixel_vector = yaw_rot.apply(pitch_rot.apply(Vector3D.plus_i()))

    return
