"""Tests for voronoiz/voronoi.py. Usage: python -m pytest voronoiz/test_voronoi.py -v"""
import unittest, random, sys, os
sys.path.insert(0, '.')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'voronoiz'))
from voronoi import generateL1Voronoi, voronoi_l1
from _voronoi_l1 import _l1_closest, _linf_halfplane


class TestL1Closest(unittest.TestCase):
    def test_horizontal(self):
        p = _l1_closest([0, 5], [10, 5], 0, 20, 0, 20)
        self.assertGreater(len(p), 2)
        for pt in p: self.assertLessEqual(pt[0], 5.0 + 1e-10)

    def test_diag_square(self):
        p = _l1_closest([2, 2], [6, 6], 0, 10, 0, 10)
        self.assertGreater(len(p), 2)

    def test_anti_square(self):
        p = _l1_closest([2, 6], [6, 2], 0, 10, 0, 10)
        self.assertGreater(len(p), 2)

    def test_duplicate_raises(self):
        with self.assertRaises(ValueError):
            _l1_closest([5, 5], [5, 5], 0, 10, 0, 10)


class TestLinfHalfplane(unittest.TestCase):
    def test_plus45(self):
        p = _linf_halfplane([0, 0], [4, 4], -1, 5, -1, 5)
        self.assertGreater(len(p), 2)

    def test_minus45(self):
        p = _linf_halfplane([0, 4], [4, 0], -1, 5, -1, 5)
        self.assertGreater(len(p), 2)


class TestVoronoiL1(unittest.TestCase):
    def test_square_pair(self):
        cells = voronoi_l1([[2, 2], [6, 6]], 0, 10, 0, 10)
        self.assertEqual(len(cells), 2)

    def test_grid4(self):
        cells = voronoi_l1([[0, 0], [10, 0], [0, 10], [10, 10]], 0, 20, 0, 20)
        self.assertEqual(len(cells), 4)


class TestGenerateL1Voronoi(unittest.TestCase):
    def test_4pt(self):
        r = generateL1Voronoi([[4, 6], [3, 10], [10, 6], [1, 2]], 30, 30)
        self.assertEqual(len(r), 4)
        for s in r:
            self.assertIn('d', s)
            self.assertIn('neighbors', s)
            self.assertGreaterEqual(len(s['polygonPoints']), 3)
            self.assertGreaterEqual(len(s['neighbors']), 1)

    def test_square_pair(self):
        r = generateL1Voronoi([[2, 2], [6, 6]], 10, 10)
        self.assertEqual(len(r), 2)

    def test_collinear3(self):
        r = generateL1Voronoi([[2, 5], [6, 5], [10, 5]], 15, 15)
        self.assertEqual(len(r), 3)

    def test_single(self):
        r = generateL1Voronoi([[5, 5]], 10, 10)
        self.assertEqual(len(r), 1)

    def test_empty(self):
        r = generateL1Voronoi([], 10, 10)
        self.assertEqual(len(r), 0)

    def test_no_self_intersect(self):
        for seed in range(10):
            with self.subTest(seed=seed):
                random.seed(seed)
                sites = [[random.randint(0, 200), random.randint(0, 200)] for _ in range(15)]
                r = generateL1Voronoi(sites, 200, 200)
                for site in r:
                    pts = site['polygonPoints']
                    n = len(pts)
                    if n < 4: continue
                    for i in range(n):
                        for j in range(i + 2, n):
                            if i == 0 and j == n - 1: continue
                            if _cross(pts[i], pts[(i+1)%n], pts[j], pts[(j+1)%n]):
                                self.fail(f"seed {seed}: self-intersect at {site['site']}")

    def test_stress(self):
        random.seed(1)
        def _rn(s): return sum(random.random() for _ in range(s))/s
        sites = [[int(_rn(2)*400), int(_rn(2)*400)] for _ in range(89)]
        r = generateL1Voronoi(sites, 400, 400)
        self.assertGreaterEqual(len(r), 1)


def _cross(a, b, c, d):
    def eq(p, q): return abs(p[0]-q[0]) < 1e-12 and abs(p[1]-q[1]) < 1e-12
    if eq(a, b) or eq(c, d): return False
    denom = (d[1]-c[1])*(b[0]-a[0]) - (d[0]-c[0])*(b[1]-a[1])
    if denom == 0: return False
    ua = ((d[0]-c[0])*(a[1]-c[1]) - (d[1]-c[1])*(a[0]-c[0])) / denom
    ub = ((b[0]-a[0])*(a[1]-c[1]) - (b[1]-a[1])*(a[0]-c[0])) / denom
    return 0 < ua < 1 and 0 < ub < 1


if __name__ == '__main__':
    unittest.main()
