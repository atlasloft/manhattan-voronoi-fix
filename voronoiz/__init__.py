import sys, os
_srcdir = os.path.dirname(os.path.abspath(__file__))
if _srcdir not in sys.path: sys.path.insert(0, _srcdir)
from _voronoi_l1 import voronoi_l1 as _voronoi_l1_raw
from voronoi import generateL1Voronoi
