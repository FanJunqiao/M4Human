# Copyright (C) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np

"""
77GH is Horizontal
79GH is Vertical
"""
R77 = np.array(
    [
        [0.999428038084798, 0.00427390091669328, -0.0335459455214323],
        [0.00166452019233972, -0.996996325502861, -0.0774309776919027],
        [0.0337761167469250, -0.0773308522179145, 0.996433195569452],
    ]
)
t77 = np.array([[-0.0522650930091782], [0.357508907988834], [0.0960279687120655]])

R79 = np.array(
    [
        [0.997214252753733, -0.0607987476962386, -0.0432116463859367],
        [-0.0630518006224704, -0.996610215186671, -0.0528445779985113],
        [0.0398522840384131, -0.0554219384733665, 0.997667381541953],
    ]
)
t79 = np.array([[-0.0648588235850897], [0.302966783204636], [0.0914051241089986]])

"""
radarHori: horizontal radar heatmap list with x- and z-axes in Camera 3D Cartesian coordinate with 3 elements
    xAxis: the x-axis in the range [-1, 4] meters
    zAxis: the z-axis in the range [0, 8] meters
radarVert: vertical radar heatmap list with y- and z-axes in Camera 3D Cartesian coordinate with 3 elements
    yAxis: the y-axis in the range [-2, 3] meters
    zAxis: the z-axis in the range [0, 8] meters
"""
x_range = [-1, 4]  # radar cartesian coordinate azimus
y_range = [-2, 3]  # radar cartesian coordinate elevation
z_range = [0, 8]  # radar cartesian coordinate range

W, H = 640, 480  # camera image size
fx, fy = 379.476, 379.476  # Focal lengths
ppx, ppy = 322.457, 241.627  # Principal point coordinates
k1, k2, k3 = -0.0566423, 0.0700418, -0.000190029  # Radial distortion coefficients
p1, p2 = 6.1314e-05, -0.0222783  # Tangential distortion coefficients
