import numpy as np
import cv2 as cv
import math

# mouse callback function
def mouse(event, x, y, flags, param):
	global p, i
	if (event == cv.EVENT_RBUTTONDOWN):
		i = max(-1, i - 1)
	if (event == cv.EVENT_LBUTTONDOWN):
		i = min(5, i + 1)
		p[i] = (x, y)
		if (i == 2):
			calc()
'''
ALGORITHM:

(1) Identify a seed set of image-to-image matches between the two images. At least four points, ^ti and ^si (i = 1, 2), are needed, though more are preferable. The feature points can be refined as shown in Section 4.1.
(2) Obtain the initial closed form solution
	(a) Compute the sun position v using (4) and shadow vanishing point vs using (5) for each view, and thus obtain two linear constraints on x using (8).
	(b) Compute the third constraint using (15).
	(c) Solve the two linear constraints in (8) and one quadratic constraint in (15) by assuming x12 = 0 and x22 = 1. This gives us x13, x23 and x33 up to an ambiguity.
(3) Obtain the refined solution
	(a) Compute the fundamental matrix F (when enough point correspondences are available) as described in Section 3.2.
	(b) Minimize the cost function in (14) using the above x13, x23, and x33 as starting points. The ambiguity in (c) is eliminated during the minimization process, leading to the correct solution.
(4) Use bundle adjustment [43] to refine further the solution.
(5) Compute the orientation of the light source using (17) and (18).

EQUATIONS:

(4)  v ~ (t1 x s1) x (t2 x s2)
(5)  vs = (H * (v'z x v')) x (vz x v)
(8)  vsT * w * vz = 0
(14) Ʃ{d1(v'i, Hvi)^2 + d2(v'i, Fvi)^2}
where d1(x,y) is the geometric image distance between the two homogeneous image points represented by x and y, and d2(x,l) is the geometric image distance from an image point x to an image line l
(15) {v1, s2; s1, a}(1) = {v1, s2; s1, a}(2)
where {Æ,Æ;Æ,Æ} denotes the cross-ratio of four points, and the superscripts indicate the images in which the cross-ratios are taken
(17) Φ = cos^-1((vzT * w * v) / (sqrt(vT * w * v) * sqrt(vzT * w * vz)))
(18) θ = cos^-1((vxT * w * vs) / (sqrt(vsT * w * vs) * sqrt(vxT * w * vx)))

OBS:

Cross product (returns 3d array): arr = np.cross(arr1, arr2)
'''

# calculates angle between two points
def angle(p1, p2, center):
	p1 = np.subtract(p1, center)
	p2 = np.subtract(p2, center)
	ang1 = np.arctan2(*p1[::-1])
	ang2 = np.arctan2(*p2[::-1])
	print(np.rad2deg((ang1 - ang2) % (2 * np.pi)))
	return (ang1 - ang2) % (2 * np.pi)

# calculates light vector
def calc():
	global light
	h = math.dist(p[0], p[1])
	s = math.dist(p[1], p[2])
	a = angle(p[0], p[2], p[1])
	light = (s * math.cos(a), s * math.sin(a), h)
	print(light)

# preparation
original = cv.imread('dataset/shadow/00122.jpg')
p = [(-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1), (-1, -1)]
i = -1
light = (0, 0, 0)
cv.namedWindow('Get Light Vector')
cv.setMouseCallback('Get Light Vector', mouse)

# main loop
while (not (cv.waitKey(20) & 0xFF == 27)):
	img = original.copy()
	if (i > -1):
		cv.circle(img, p[0], 10, (255,   0,   0), -1)
	if (i > 0):
		cv.circle(img, p[1], 10, (  0, 255,   0), -1)
	if (i > 1):
		cv.circle(img, p[2], 10, (  0,   0, 255), -1)
	if (i > 2):
		cv.circle(img, p[3], 10, (  0, 255, 255), -1)
	if (i > 3):
		cv.circle(img, p[4], 10, (255,   0, 255), -1)
	if (i > 4):
		cv.circle(img, p[5], 10, (255, 255,   0), -1)
	cv.imshow('Get Light Vector', img)

# end
cv.destroyAllWindows()
