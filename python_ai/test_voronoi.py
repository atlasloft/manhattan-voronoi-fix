import unittest
import random
import sys
import math
sys.path.insert(0, '.')
from voronoi import (
    bisectorIntersection, findL1Bisector,
    samePoint, generateVoronoiPoints, cleanData, generateL1Voronoi,
    distance, angle, segmentIntersection,
)


class TestSegmentIntersection(unittest.TestCase):

    def test_horizontal_cross_vertical(self):
        pt = segmentIntersection([[0, 0], [4, 0]], [[2, 2], [2, -2]])
        self.assertTrue(samePoint(pt, [2, 0]))

    def test_two_diagonals(self):
        pt = segmentIntersection([[0, 0], [4, 4]], [[4, 0], [0, 4]])
        self.assertTrue(samePoint(pt, [2, 2]))

    def test_parallel_verticals(self):
        pt = segmentIntersection([[0, 0], [0, 4]], [[2, 0], [2, 4]])
        self.assertIsNone(pt)

    def test_non_intersecting(self):
        pt = segmentIntersection([[0, 0], [1, 0]], [[2, 1], [3, 1]])
        self.assertFalse(pt)


class TestBisectorIntersection(unittest.TestCase):

    def _s(self, x, y):
        return {'site': [x, y], 'bisectors': []}

    def test_normal_shared_site(self):
        A = self._s(4, 6)
        B = self._s(3, 10)
        C1 = self._s(10, 6)
        C2 = self._s(10, 6)
        bAC = findL1Bisector(A, C1, 30, 30)
        bBC = findL1Bisector(B, C2, 30, 30)
        P = bisectorIntersection(bAC, bBC)
        self.assertIsInstance(P, list)
        self.assertEqual(len(P), 2)

    def test_non_shared_bisectors(self):
        A = self._s(0, 0)
        B = self._s(10, 0)
        C = self._s(0, 20)
        D = self._s(10, 20)
        bAB = findL1Bisector(A, B, 30, 30)
        bCD = findL1Bisector(C, D, 30, 30)
        P = bisectorIntersection(bAB, bCD)
        self.assertFalse(P)

    def test_no_sites_key(self):
        zline = {'points': [[-4, 4], [200, 4]]}
        C = self._s(-8, -2)
        bBC = findL1Bisector(self._s(-1, 1), C, 200, 200)
        P = bisectorIntersection(zline, bBC)
        self.assertTrue(P is False or isinstance(P, list))


class TestFindL1Bisector(unittest.TestCase):

    def _s(self, x, y):
        return {'site': [x, y], 'bisectors': []}

    def test_horizontal_sites(self):
        A = self._s(0, 5)
        B = self._s(10, 5)
        bisector = findL1Bisector(A, B, 100, 100)
        self.assertEqual(bisector['up'], True)
        self.assertGreaterEqual(len(bisector['points']), 2)

    def test_vertical_sites(self):
        A = self._s(5, 0)
        B = self._s(5, 10)
        bisector = findL1Bisector(A, B, 100, 100)
        self.assertEqual(bisector['up'], False)
        self.assertGreaterEqual(len(bisector['points']), 2)

    def test_diagonal_slope_positive(self):
        A = self._s(0, 0)
        B = self._s(4, 4)
        bisector = findL1Bisector(A, B, 200, 200)
        self.assertIn('points', bisector)
        self.assertGreaterEqual(len(bisector['points']), 2)

    def test_diagonal_slope_negative(self):
        A = self._s(0, 4)
        B = self._s(4, 0)
        bisector = findL1Bisector(A, B, 200, 200)
        self.assertIn('points', bisector)
        self.assertGreaterEqual(len(bisector['points']), 2)

    def test_duplicate_raises(self):
        A = self._s(5, 5)
        B = self._s(5, 5)
        with self.assertRaises(ValueError):
            findL1Bisector(A, B, 100, 100)

    def test_no_consecutive_duplicates(self):
        A = self._s(0, 0)
        B = self._s(4, 10)
        b = findL1Bisector(A, B, 100, 100)
        for k in range(len(b['points']) - 1):
            self.assertFalse(samePoint(b['points'][k], b['points'][k + 1]))


class TestCleanData(unittest.TestCase):

    def test_nudge(self):
        data = [[4, 6], [6, 4]]
        cleaned = cleanData([list(p) for p in data])
        # Points on a square (|dx|==|dy|) should be nudged
        self.assertNotEqual(cleaned[1], [6, 4])


class TestGenerateVoronoiPoints(unittest.TestCase):

    def test_basic(self):
        pts = [[4, 6], [3, 10], [10, 6], [1, 2]]
        result = generateVoronoiPoints(pts, 30, 30, distance)
        self.assertEqual(len(result), 900)


class TestGenerateL1Voronoi(unittest.TestCase):

    def test_four_sites(self):
        sites = [[4, 6], [3, 10], [10, 6], [1, 2]]
        result = generateL1Voronoi(sites, 30, 30, nudgeData=True)
        self.assertEqual(len(result), 4)
        for site in result:
            self.assertIn('d', site)
            self.assertIn('neighbors', site)
            self.assertIn('polygonPoints', site)
            self.assertGreaterEqual(len(site['polygonPoints']), 3)
            self.assertGreaterEqual(len(site['neighbors']), 1)

    def test_16_random_sites(self):
        random.seed(42)
        sites = [[int(random.random() * 400), int(random.random() * 400)]
                 for _ in range(16)]
        result = generateL1Voronoi(sites, 400, 400, nudgeData=True)
        self.assertEqual(len(result), 16)
        for site in result:
            self.assertGreaterEqual(len(site['neighbors']), 1)
            self.assertGreaterEqual(len(site['polygonPoints']), 3)

    def test_no_nudge(self):
        sites = [[4, 6], [3, 10], [10, 6], [1, 2]]
        result = generateL1Voronoi(sites, 30, 30, nudgeData=False)
        self.assertEqual(len(result), 4)

    def test_duplicate_points(self):
        sites = [[4, 6], [4, 6], [10, 6], [1, 2]]
        result = generateL1Voronoi(sites, 30, 30, nudgeData=True)
        self.assertLessEqual(len(result), 4)


class TestRandomStress(unittest.TestCase):

    @staticmethod
    def _random_normal(sharpness):
        return sum(random.random() for _ in range(sharpness)) / sharpness

    def _gen_sites(self, seed, n):
        random.seed(seed)
        return [[int(self._random_normal(2) * 400),
                 int(self._random_normal(2) * 400)] for _ in range(n)]

    def test_89_sites(self):
        sites = self._gen_sites(1, 89)
        result = generateL1Voronoi(sites, 400, 400)
        self.assertGreaterEqual(len(result), 1)
        for site in result:
            self.assertGreaterEqual(len(site['neighbors']), 1)
            self.assertGreaterEqual(len(site['polygonPoints']), 3)
            self.assertIn('d', site)
            self.assertTrue(site['d'].startswith('M '))
            self.assertTrue(site['d'].endswith(' Z'))

    def test_multi_seed_stress(self):
        for seed in range(20):
            with self.subTest(seed=seed):
                sites = self._gen_sites(seed, 67)
                result = generateL1Voronoi(sites, 400, 400)
                for site in result:
                    self.assertGreaterEqual(len(site['neighbors']), 1,
                                            f"seed {seed}: site {site['site']} has no neighbors")
                    self.assertGreaterEqual(len(site['polygonPoints']), 3,
                                            f"seed {seed}: site {site['site']} has <3 polygon points")
                for site in result:
                    for bisector in site['bisectors']:
                        pts = bisector['points']
                        self.assertGreaterEqual(len(pts), 2,
                                                f"seed {seed}: bisector has <2 points")
                        for k in range(len(pts) - 1):
                            self.assertFalse(samePoint(pts[k], pts[k + 1]),
                                             f"seed {seed}: zero-length segment in bisector")

    def test_polygon_simple(self):
        for seed in range(10):
            with self.subTest(seed=seed):
                sites = self._gen_sites(seed, 20)
                result = generateL1Voronoi(sites, 400, 400)
                for site in result:
                    pts = site['polygonPoints']
                    n = len(pts)
                    if n < 3:
                        continue
                    # Check each pair of non-adjacent edges
                    for i in range(n):
                        i_next = (i + 1) % n
                        for j in range(i + 2, n):
                            if i == 0 and j == n - 1:
                                continue  # adjacent via wrap-around
                            j_next = (j + 1) % n
                            seg1 = (pts[i], pts[i_next])
                            seg2 = (pts[j], pts[j_next])
                            if self._segments_cross(seg1[0], seg1[1],
                                                    seg2[0], seg2[1]):
                                self.fail(
                                    f"seed {seed}: self-intersecting polygon at {site['site']}"
                                )

    @staticmethod
    def _segments_cross(p0, p1, p2, p3):
        """Check if segments (p0,p1) and (p2,p3) cross in their interiors."""
        # Skip zero-length segments (duplicate consecutive points are filtered)
        if samePoint(p0, p1) or samePoint(p2, p3):
            return False
        d1 = (p1[0] - p0[0], p1[1] - p0[1])
        d2 = (p3[0] - p2[0], p3[1] - p2[1])
        denom = d2[1] * d1[0] - d2[0] * d1[1]
        if denom == 0:
            return False
        ua = (d2[0] * (p0[1] - p2[1]) - d2[1] * (p0[0] - p2[0])) / denom
        ub = (d1[0] * (p0[1] - p2[1]) - d1[1] * (p0[0] - p2[0])) / denom
        return 0 < ua < 1 and 0 < ub < 1

    def test_bisector_no_overlapping_points(self):
        for seed in range(10):
            with self.subTest(seed=seed):
                sites = self._gen_sites(seed, 30)
                result = generateL1Voronoi(sites, 400, 400)
                for site in result:
                    for b in site['bisectors']:
                        pts = b['points']
                        for i in range(len(pts)):
                            for j in range(i + 2, len(pts)):
                                if i == 0 and j == len(pts) - 1:
                                    continue
                                self.assertFalse(
                                    samePoint(pts[i], pts[j]),
                                    f"seed {seed}: overlapping points in bisector"
                                )


class TestAngle(unittest.TestCase):

    def test_zero_degrees(self):
        a = angle([0, 0], [10, 0])
        self.assertAlmostEqual(a, 0.0, places=5)

    def test_90_degrees(self):
        a = angle([0, 0], [0, 10])
        self.assertAlmostEqual(a, math.pi / 2, places=5)

    def test_180_degrees(self):
        a = angle([0, 0], [-10, 0])
        self.assertAlmostEqual(a, math.pi, places=5)

    def test_negative_angle_normalized(self):
        a = angle([0, 0], [10, -10])
        self.assertGreaterEqual(a, 0)
        self.assertLess(a, 2 * math.pi)


if __name__ == '__main__':
    unittest.main()
