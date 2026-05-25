"""
Check all 56 voronoiz test results for self-intersecting polygons.
Usage: python voronoiz_test/check_self_intersect.py
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from voronoiz.voronoi import generateL1Voronoi

SEED = 20250526


def segments_cross(a, b, c, d):
    """True if segments (a,b) and (c,d) intersect strictly in their interiors."""
    def eq(p, q):
        return abs(p[0] - q[0]) < 1e-12 and abs(p[1] - q[1]) < 1e-12
    if eq(a, b) or eq(c, d):
        return False
    denom = (d[1] - c[1]) * (b[0] - a[0]) - (d[0] - c[0]) * (b[1] - a[1])
    if denom == 0:
        return False
    ua = ((d[0] - c[0]) * (a[1] - c[1]) - (d[1] - c[1]) * (a[0] - c[0])) / denom
    ub = ((b[0] - a[0]) * (a[1] - c[1]) - (b[1] - a[1]) * (a[0] - c[0])) / denom
    return 0 < ua < 1 and 0 < ub < 1


def check_self_intersection(result):
    """Return list of (site_idx, edge_pairs) for self-intersecting polygons."""
    errors = []
    for idx, site in enumerate(result):
        pts = site['polygonPoints']
        n = len(pts)
        if n < 4:
            continue
        found = False
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                if segments_cross(pts[i], pts[(i + 1) % n],
                                  pts[j], pts[(j + 1) % n]):
                    errors.append({
                        'site_index': idx,
                        'site': site['site'],
                        'edges': ((i, (i + 1) % n), (j, (j + 1) % n)),
                    })
                    found = True
                    break
            if found:
                break
    return errors


# ── generators ───────────────────────────────────────────────

def gen_random(n, w, h, seed):
    r = random.Random(seed)
    return [[r.randint(0, w), r.randint(0, h)] for _ in range(n)]

def gen_grid(rows, cols, w, h):
    xs = [round(w * c / max(cols - 1, 1)) for c in range(cols)]
    ys = [round(h * r / max(rows - 1, 1)) for r in range(rows)]
    return [[x, y] for y in ys for x in xs]

def gen_diagonal(n, w, h, seed):
    r = random.Random(seed)
    s = min(w, h)
    return [[x, x] for x in sorted(r.sample(range(s + 1), min(n, s + 1)))]

def gen_line(n, w, h, seed):
    r = random.Random(seed)
    return [[x, h // 2] for x in sorted(r.sample(range(w + 1), min(n, w + 1)))]

def gen_clustered(n, w, h, seed, clusters=3):
    r = random.Random(seed)
    pts = []
    per = n // clusters
    for _ in range(clusters):
        cx, cy = r.randint(w // 4, 3 * w // 4), r.randint(h // 4, 3 * h // 4)
        for _ in range(per):
            pts.append([max(0, min(w, cx + r.randint(-w // 8, w // 8))),
                        max(0, min(h, cy + r.randint(-h // 8, h // 8)))])
    return pts[:n]


# ── run ──────────────────────────────────────────────────────

cases = []

for slug, sites, w, h in [
    ("basic_diag_pair",        [[2, 2], [6, 6]], 10, 10),
    ("basic_anti_pair",        [[2, 6], [6, 2]], 10, 10),
    ("basic_mix_4pt",          [[4, 6], [3, 10], [10, 6], [1, 2]], 30, 30),
    ("basic_grid_2x2",         [[0, 0], [10, 0], [0, 10], [10, 10]], 20, 20),
    ("basic_align_5pt",        [[5, 5], [10, 10], [15, 5], [5, 15], [15, 15]], 20, 20),
    ("basic_collinear_3pt",    [[2, 5], [6, 5], [10, 5]], 15, 15),
    ("basic_fat_region",       [[2, 2], [6, 6], [0, 8]], 10, 10),
]:
    cases.append((slug, sites, w, h))

for i, (n, w, h) in enumerate([(10, 50, 50), (30, 100, 100), (50, 200, 200), (100, 400, 400)]):
    cases.append((f"rand_n{n}", gen_random(n, w, h, SEED + i), w, h))

for r, c, w, h in [(4, 4, 40, 40), (6, 6, 60, 60), (8, 3, 80, 30), (5, 5, 50, 50)]:
    cases.append((f"grid_{r}x{c}", gen_grid(r, c, w, h), w, h))

for i, (n, w, h) in enumerate([(5, 50, 50), (10, 100, 100), (20, 200, 200), (40, 400, 400)]):
    cases.append((f"line_n{n}", gen_line(n, w, h, SEED + 100 + i), w, h))

for i, (n, w, h) in enumerate([(5, 50, 50), (10, 100, 100), (15, 150, 150), (20, 200, 200)]):
    cases.append((f"diag_n{n}", gen_diagonal(n, w, h, SEED + 200 + i), w, h))

for i, (n, cl) in enumerate([(15, 3), (30, 5), (60, 6)]):
    cases.append((f"cluster_n{n}", gen_clustered(n, 200, 200, SEED + 300 + i, cl), 200, 200))

for rnd in range(30):
    cases.append((f"round_{rnd:02d}", gen_random(20, 150, 150, SEED + 1000 + rnd), 150, 150))


print(f"Checking {len(cases)} test cases for self-intersecting polygons...\n")

total_errors = 0
total_polygons = 0

for slug, sites, w, h in cases:
    try:
        result = generateL1Voronoi(sites, w, h)
    except Exception as e:
        print(f"  {slug}: CRASH — {e}")
        continue

    errors = check_self_intersection(result)
    total_polygons += len(result)
    total_errors += len(errors)

    if errors:
        print(f"  FAIL  {slug}: {len(errors)} self-intersecting polygon(s)")
        for e in errors:
            print(f"         site[{e['site_index']}] {e['site']}  "
                  f"edges {e['edges'][0]} × {e['edges'][1]}")
    else:
        pass  # clean

print(f"\n{'='*50}")
print(f"Result: {total_errors} self-intersections / {total_polygons} polygons")
print(f"Status: {'PASS' if total_errors == 0 else 'FAIL'}")
