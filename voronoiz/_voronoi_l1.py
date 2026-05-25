# Copyright © 2021 Warren Weckesser


# 代码基于voronoiz项目以及切比雪夫距离相关原理，版权属于原作者。


"""
`voronoi_l1` creates Voronoi cells (polygons) for a set of points
in the plane using the L1 metric (also known as the city-block metric,
the Manhattan metric, or the taxicab metrix). The function
requires the Shapely library (https://pypi.org/project/Shapely/).
"""

import math
import numpy as np


def _linf_halfplane(p0, p1, xmin, xmax, ymin, ymax):
    """
    L∞ (Chebyshev) bisector half-plane polygon for two points on a ±45° line.

    When two points lie on a ±45° line, the L1 bisector degenerates into a
    2-d region.  The L∞ bisector, however, is always a single line that
    passes through that degenerate band — a natural deterministic fallback.

    Returns a numpy array of polygon vertices bounding the set of points
    where the L∞ distance to p0 is less than or equal to the L∞ distance
    to p1, clipped to the bounding box.
    """
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0

    if dx == dy:
        # +45° line: L∞ bisector is  x + y = C
        use_sum = True
        C = (x0 + y0 + x1 + y1) / 2.0
        p0_val = x0 + y0
    else:
        # -45° line (dx == -dy): L∞ bisector is  x - y = C
        use_sum = False
        C = (x0 - y0 + x1 - y1) / 2.0
        p0_val = x0 - y0

    # Which side of the bisector is p0 on?
    side_leq = p0_val < C   # True  → p0's half-plane is value ≤ C
                             # False → p0's half-plane is value ≥ C

    corners = np.array([
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
    ])

    if use_sum:
        values = corners[:, 0] + corners[:, 1]        # x + y
    else:
        values = corners[:, 0] - corners[:, 1]        # x - y

    inside = values <= C if side_leq else values >= C

    vertices = []
    for i in range(4):
        j = (i + 1) % 4
        if inside[i]:
            vertices.append(corners[i])
        if inside[i] != inside[j]:
            pi, pj = corners[i], corners[j]
            if use_sum:
                num = C - (pi[0] + pi[1])
                denom = (pj[0] - pi[0]) + (pj[1] - pi[1])
            else:
                num = C - (pi[0] - pi[1])
                denom = (pj[0] - pi[0]) - (pj[1] - pi[1])
            if abs(denom) > 1e-15:
                t = num / denom
                # Only add if the intersection is strictly between corners
                # (not coincident with a corner, which was already added)
                if 1e-12 < t < 1 - 1e-12:
                    vertices.append(pi + t * (pj - pi))

    p = np.array(vertices)
    if len(p) > 0:
        p = np.vstack([p, p[0]])
    return p


def _l1_closest(p0, p1, xmin, xmax, ymin, ymax):
    """
    Return the vertices of a polygon bounding the set of points where
    the L1 distance to p0 is less than the L1 distance to p1.  The
    region is clipped to the bounding box defined by (xmin, xmax,
    ymin, ymax).

    An error is raised if p0 == p1.  When p0 and p1 lie on a ±45 degree
    line (L1 degeneracy), the L∞ (Chebyshev) bisector is used as a
    deterministic fallback.

    Examples
    --------
    >>> _l1_closest(p0=[0, 0], p1=[4, 2], xmin=-1, xmax=5, ymin=-1, ymax=3)
    array([[ 3.,  0.],
           [ 1.,  2.],
           [ 1.,  3.],
           [-1.,  3.],
           [-1., -1.],
           [ 3., -1.],
           [ 3.,  0.]])
    """
    x0, y0 = p0
    x1, y1 = p1
    width = abs(x1 - x0)
    height = abs(y1 - y0)
    if width == height == 0:
        raise ValueError('points must not be equal')
    if width == height:
        # L1 bisector is degenerate (a region, not a line).
        # Fall back to the L∞ (Chebyshev) bisector, which for points
        # on a ±45° line is a single clean line through the degenerate
        # band.  This is a deterministic tie-breaking choice: when L1
        # cannot decide, use L∞.
        return _linf_halfplane(p0, p1, xmin, xmax, ymin, ymax)

    flip = False
    if width > height:
        xmin, xmax, ymin, ymax = ymin, ymax, xmin, xmax
        x0, y0, x1, y1 = y0, x0, y1, x1
        width, height = height, width
        flip = True

    if width == 0:
        mid = 0.5*(y0 + y1)
        if y0 < y1:
            p = np.array([[xmin, mid],
                          [xmax, mid],
                          [xmax, ymin],
                          [xmin, ymin],
                          [xmin, mid]])
        else:
            p = np.array([[xmin, mid],
                          [xmin, ymax],
                          [xmax, ymax],
                          [xmax, mid],
                          [xmin, mid]])
    else:
        distance = width + height
        s = math.copysign(1, y1 - y0)
        pa = np.array([x0, y0 + s*0.5*(distance)])
        pb = np.array([x1, y1 - s*0.5*(distance)])
        if x0 < x1 and y0 < y1:
            p = np.array([pa,
                          pb,
                          [xmax, pb[1]],
                          [xmax, ymin],
                          [xmin, ymin],
                          [xmin, pa[1]],
                          pa])
        elif x0 < x1 and y0 > y1:
            p = np.array([pa,
                          pb,
                          [xmax, pb[1]],
                          [xmax, ymax],
                          [xmin, ymax],
                          [xmin, pa[1]],
                          pa])
        elif x0 > x1 and y0 < y1:
            p = np.array([pa,
                          pb,
                          [xmin, pb[1]],
                          [xmin, ymin],
                          [xmax, ymin],
                          [xmax, pa[1]],
                          pa])
        else:
            # x0 > x1 and y0 > y1
            p = np.array([pa,
                          pb,
                          [xmin, pb[1]],
                          [xmin, ymax],
                          [xmax, ymax],
                          [xmax, pa[1]],
                          pa])
    if flip:
        p = p[:, ::-1]
    return p


def voronoi_l1(points, xmin, xmax, ymin, ymax):
    """
    Compute Voronoi cells using the L1 metric.

    The L1 metric is also known as the "city block" metric, the
    taxicab metric, or the Manhattan metric.

    The cells (polygons represented as arrays of 2-d points) are clipped
    to the bounding box defined by `xmin`, `xmax`, `ymin`, `ymax`.

    The return value is a list of numpy arrays.  The i-th list
    has shape (n[i], 2), where n[i] is the number of vertices
    in the Voronoi cell around `points[i]`.

    There must be no duplicate points.  When two points lie on a common
    45 or -45 degree line, the L1 bisector is degenerate (a 2-d region).
    In that case, the L∞ (Chebyshev) bisector is used as a deterministic
    fallback to produce a well-defined cell boundary.
    """
    from shapely.geometry import Polygon

    cells = []
    for i0 in range(len(points)):
        p0 = points[i0]
        poly = []
        for i1 in range(len(points)):
            if i1 == i0:
                continue
            p1 = points[i1]
            p = _l1_closest(p0, p1, xmin, xmax, ymin, ymax)
            poly.append(Polygon(p))

        region = poly[0]
        for r in poly[1:]:
            region = region.intersection(r)

        cells.append(np.column_stack(region.exterior.xy))
    return cells
