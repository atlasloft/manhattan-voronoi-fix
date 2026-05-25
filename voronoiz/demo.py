"""Demo: L1 Voronoi via half-plane intersection. Usage: python voronoiz/demo.py"""
import random, colorsys
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import sys, os
from voronoiz.voronoi import generateL1Voronoi


def rn(s): return sum(random.random() for _ in range(s)) / s


def cell_color(i, n):
    h = (i * 0.618033988749895) % 1.0
    return colorsys.hsv_to_rgb(h, 0.5, 0.95)


def main():
    w = h = 400
    random.seed(1)
    sites = [[int(rn(2) * w), int(rn(2) * h)] for _ in range(89)]
    result = generateL1Voronoi(sites, w, h)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    drawn = set()
    for i, site in enumerate(result):
        pts = [(x, h - y) for x, y in site['polygonPoints']]
        ax1.add_patch(Polygon(pts, closed=True, edgecolor='black',
                              facecolor=cell_color(i, len(result)),
                              linewidth=0.5, alpha=0.8))
        ax1.plot(site['site'][0], h - site['site'][1], 'ko', markersize=3)
        for b in site['bisectors']:
            if id(b) in drawn: continue
            drawn.add(id(b))
            bp = [(x, h - y) for x, y in b['points']]
            ax2.plot([p[0] for p in bp], [p[1] for p in bp], color='#333', linewidth=0.5, alpha=0.5)
        ax2.plot(site['site'][0], h - site['site'][1], 'ko', markersize=2)

    ax1.set(xlim=(0, w), ylim=(0, h), aspect='equal', title='Cells')
    ax2.set(xlim=(0, w), ylim=(0, h), aspect='equal', title='Bisectors')
    plt.tight_layout()
    plt.savefig('voronoi_demo.png', dpi=150)
    plt.close()
    print(f"Saved: voronoiz/voronoi_demo.png ({len(result)} sites)")


if __name__ == "__main__":
    main()
