"""Run all 56 test cases through voronoiz/voronoi.py, output PNG to voronoiz_test/."""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from voronoiz.voronoi import generateL1Voronoi
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 20250526

def gen_random(n, w, h, seed):
    r = random.Random(seed)
    return [[r.randint(0, w), r.randint(0, h)] for _ in range(n)]
def gen_grid(rows, cols, w, h):
    xs = [round(w*c/max(cols-1,1)) for c in range(cols)]
    ys = [round(h*r/max(rows-1,1)) for r in range(rows)]
    return [[x, y] for y in ys for x in xs]
def gen_diagonal(n, w, h, seed):
    r = random.Random(seed); s = min(w, h)
    return [[x, x] for x in sorted(r.sample(range(s+1), min(n, s+1)))]
def gen_line(n, w, h, seed):
    r = random.Random(seed)
    return [[x, h//2] for x in sorted(r.sample(range(w+1), min(n, w+1)))]
def gen_clustered(n, w, h, seed, clusters=3):
    r = random.Random(seed); pts = []; per = n // clusters
    for _ in range(clusters):
        cx, cy = r.randint(w//4, 3*w//4), r.randint(h//4, 3*h//4)
        for _ in range(per):
            pts.append([max(0, min(w, cx+r.randint(-w//8, w//8))),
                        max(0, min(h, cy+r.randint(-h//8, h//8)))])
    return pts[:n]

cases = []
for slug, sites, w, h in [
    ("basic_diag_pair", [[2,2],[6,6]], 10, 10),
    ("basic_anti_pair", [[2,6],[6,2]], 10, 10),
    ("basic_mix_4pt", [[4,6],[3,10],[10,6],[1,2]], 30, 30),
    ("basic_grid_2x2", [[0,0],[10,0],[0,10],[10,10]], 20, 20),
    ("basic_align_5pt", [[5,5],[10,10],[15,5],[5,15],[15,15]], 20, 20),
    ("basic_collinear_3pt", [[2,5],[6,5],[10,5]], 15, 15),
    ("basic_fat_region", [[2,2],[6,6],[0,8]], 10, 10),
]: cases.append((slug, sites, w, h))

for i, (n, w, h) in enumerate([(10,50,50),(30,100,100),(50,200,200),(100,400,400)]):
    cases.append((f"rand_n{n}", gen_random(n,w,h,SEED+i), w, h))
for r, c, w, h in [(4,4,40,40),(6,6,60,60),(8,3,80,30),(5,5,50,50)]:
    cases.append((f"grid_{r}x{c}", gen_grid(r,c,w,h), w, h))
for i, (n, w, h) in enumerate([(5,50,50),(10,100,100),(20,200,200),(40,400,400)]):
    cases.append((f"line_n{n}", gen_line(n,w,h,SEED+100+i), w, h))
for i, (n, w, h) in enumerate([(5,50,50),(10,100,100),(15,150,150),(20,200,200)]):
    cases.append((f"diag_n{n}", gen_diagonal(n,w,h,SEED+200+i), w, h))
for i, (n, cl) in enumerate([(15,3),(30,5),(60,6)]):
    cases.append((f"cluster_n{n}", gen_clustered(n,200,200,SEED+300+i,cl), 200, 200))
for rnd in range(30):
    cases.append((f"round_{rnd:02d}", gen_random(20,150,150,SEED+1000+rnd), 150, 150))

print(f"Rendering {len(cases)} cases...")
ok = crash = 0
for idx, (slug, sites, w, h) in enumerate(cases):
    try:
        r = generateL1Voronoi(sites, w, h)
        fw = max(4, min(8, w/40)); fh = fw * (h/w) + 0.3
        fig, ax = plt.subplots(figsize=(fw, fh))
        ax.set(xlim=(-0.5, w+0.5), ylim=(-0.5, h+0.5), aspect='equal')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
        n = len(r)
        colors = plt.cm.Set3(np.linspace(0, 1, max(n, 1)))
        patches = [MplPolygon(s['polygonPoints'], closed=True) for s in r if len(s['polygonPoints'])>=3]
        pc = PatchCollection(patches, alpha=0.55, edgecolor='#444', linewidth=0.6)
        pc.set_facecolor([colors[i%len(colors)] for i in range(len(patches))])
        ax.add_collection(pc)
        drawn = set()
        for s in r:
            for b in s['bisectors']:
                if id(b) in drawn: continue
                drawn.add(id(b))
                if len(b['points'])>=2:
                    xs, ys = zip(*b['points']); ax.plot(xs, ys, 'k-', lw=0.3, alpha=0.3)
        pts = np.array(sites)
        ax.scatter(pts[:,0], pts[:,1], c='black', s=25, zorder=10, edgecolors='white', lw=0.6)
        plt.tight_layout(pad=0.1)
        fig.savefig(os.path.join(OUT, f"{slug}.png"), dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig); ok += 1
    except Exception as e:
        crash += 1; print(f"  {slug}: {e}")
    if (idx+1)%10 == 0: print(f"  {idx+1}/{len(cases)}...")

print(f"\nDone: {ok} OK, {crash} errors = {len(cases)} files")
print(f"Output: {OUT}/")
