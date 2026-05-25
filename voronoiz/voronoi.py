"""Compatibility wrapper around _voronoi_l1."""
import sys, os
_srcdir = os.path.dirname(os.path.abspath(__file__))
if _srcdir not in sys.path: sys.path.insert(0, _srcdir)

import numpy as np
from shapely.geometry import Polygon
from _voronoi_l1 import voronoi_l1, _l1_closest, _linf_halfplane


def _cell_to_polygon_points(cell):
    pts = cell.T.tolist()
    result = [[x, y] for x, y in zip(pts[0], pts[1])]
    if len(result) >= 2 and result[0] == result[-1]:
        result.pop()
    return result


def _compute_bisectors(cells, sites):
    n = len(cells)
    bisectors_by_site = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            poly_i = Polygon(cells[i])
            poly_j = Polygon(cells[j])
            boundary = poly_i.intersection(poly_j)

            if boundary.is_empty:
                continue

            if boundary.geom_type == 'LineString':
                coords = list(boundary.coords)
            elif boundary.geom_type == 'MultiLineString':
                coords = []
                for line in boundary.geoms:
                    coords.extend(list(line.coords))
            elif boundary.geom_type == 'Point':
                continue
            else:
                continue

            if len(coords) < 2:
                continue

            pts = [[x, y] for x, y in coords]
            bisector = {
                'sites': [{'site': sites[i]}, {'site': sites[j]}],
                'points': pts,
                'up': None,
                'intersections': [],
                'compound': False,
            }
            bisectors_by_site[i].append(bisector)
            bisectors_by_site[j].append(bisector)

    return bisectors_by_site


def _compute_neighbors(bisectors, site_coord):
    neighbors = []
    for b in bisectors:
        for s in b['sites']:
            if s['site'] != site_coord and s['site'] not in neighbors:
                neighbors.append(s['site'])
    return neighbors


def generateL1Voronoi(sitePoints, width, height, nudgeData=True):
    """
    Generate L1 Voronoi diagram via half-plane intersection.

    nudgeData is ignored — |dx|=|dy| degeneracy is handled internally
    by L∞ (Chebyshev) bisector fallback.

    Returns list of site dicts: {site, bisectors, polygonPoints, d, neighbors}
    """
    seen = set()
    unique = []
    for p in sitePoints:
        key = (p[0], p[1])
        if key not in seen:
            seen.add(key)
            unique.append(list(p))
    sites = unique

    if len(sites) == 0:
        return []
    if len(sites) == 1:
        corners = [[0, 0], [width, 0], [width, height], [0, height]]
        return [{
            'site': sites[0], 'bisectors': [],
            'polygonPoints': corners,
            'd': 'M 0 0 L {} 0 L {} {} L 0 {} Z'.format(width, width, height, height),
            'neighbors': [],
        }]

    cells = voronoi_l1(sites, 0, width, 0, height)
    bisectors_by_site = _compute_bisectors(cells, sites)

    result = []
    for i, coord in enumerate(sites):
        poly_pts = _cell_to_polygon_points(cells[i])
        bisectors = bisectors_by_site[i]
        neighbors = _compute_neighbors(bisectors, coord)
        d = 'M ' + ' L '.join(f'{x} {y}' for x, y in poly_pts) + ' Z'
        result.append({
            'site': coord, 'bisectors': bisectors,
            'polygonPoints': poly_pts, 'd': d, 'neighbors': neighbors,
        })
    return result
