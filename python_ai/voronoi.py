"""
Manhattan (L1) Voronoi diagram generator.

Faithful Python port of Lee & Wong's divide-and-conquer algorithm
from the JavaScript implementation by Joe Dragovich.
"""

import math


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def samePoint(P1, P2):
    """Check if two coordinate pairs are identical."""
    return P1[0] == P2[0] and P1[1] == P2[1]


def _dedup_consecutive(pts):
    """Remove consecutive duplicate points from a list."""
    result = []
    for pt in pts:
        if not result or not samePoint(result[-1], pt):
            result.append(pt)
    return result


def distance(P1, P2):
    """Manhattan (L1) distance between two points."""
    return abs(P1[0] - P2[0]) + abs(P1[1] - P2[1])


def angle(P1, P2):
    """Angle from P1 to P2, normalized to [0, 2*PI)."""
    a = math.atan2(P2[1] - P1[1], P2[0] - P1[0])
    if a < 0:
        a = 2.0 * math.pi + a
    return a


# ---------------------------------------------------------------------------
# Geometry: segment and bisector intersection
# ---------------------------------------------------------------------------

def segmentIntersection(L1, L2):
    """Find the intersection point of two line segments, or None/False."""
    denom = ((L2[1][1] - L2[0][1]) * (L1[1][0] - L1[0][0])
             - (L2[1][0] - L2[0][0]) * (L1[1][1] - L1[0][1]))

    if denom == 0:
        return None

    ua = ((L2[1][0] - L2[0][0]) * (L1[0][1] - L2[0][1])
          - (L2[1][1] - L2[0][1]) * (L1[0][0] - L2[0][0])) / denom
    ub = ((L1[1][0] - L1[0][0]) * (L1[0][1] - L2[0][1])
          - (L1[1][1] - L1[0][1]) * (L1[0][0] - L2[0][0])) / denom

    if not (0 <= ua <= 1 and 0 <= ub <= 1):
        return False

    return [
        L1[0][0] + ua * (L1[1][0] - L1[0][0]),
        L1[0][1] + ua * (L1[1][1] - L1[0][1]),
    ]


def bisectorIntersection(B1, B2):
    """Find the intersection point of two bisectors, or False."""
    if B1 is B2:
        return False

    for i in range(len(B1['points']) - 1):
        for j in range(len(B2['points']) - 1):
            intersect = segmentIntersection(
                [B1['points'][i], B1['points'][i + 1]],
                [B2['points'][j], B2['points'][j + 1]],
            )
            if isinstance(intersect, list):
                return intersect

    return False


# ---------------------------------------------------------------------------
# Bisector generation
# ---------------------------------------------------------------------------

def findL1Bisector(P1, P2, width, height):
    """Generate the L1 (Manhattan) bisector between two sites."""

    xDistance = P1['site'][0] - P2['site'][0]
    yDistance = P1['site'][1] - P2['site'][1]

    if samePoint(P1['site'], P2['site']):
        raise ValueError(
            f"Duplicate point: Points {P1} and {P2} are duplicates."
        )

    midpoint = [
        (P1['site'][0] + P2['site'][0]) / 2.0,
        (P1['site'][1] + P2['site'][1]) / 2.0,
    ]

    if abs(xDistance) == 0:
        vertexes = [[0, midpoint[1]], [width, midpoint[1]]]
        bisector = {
            'sites': [P1, P2],
            'up': False,
            'points': vertexes,
            'intersections': [],
            'compound': False,
        }
        return bisector

    if abs(yDistance) == 0:
        vertexes = [[midpoint[0], 0], [midpoint[0], height]]
        bisector = {
            'sites': [P1, P2],
            'up': True,
            'points': vertexes,
            'intersections': [],
            'compound': False,
        }
        return bisector

    slope = -1.0 if yDistance / xDistance > 0 else 1.0
    intercept = midpoint[1] - midpoint[0] * slope

    if abs(xDistance) >= abs(yDistance):
        vertexes = [
            [(P1['site'][1] - intercept) / slope, P1['site'][1]],
            [(P2['site'][1] - intercept) / slope, P2['site'][1]],
        ]
        up = True
    else:
        vertexes = [
            [P1['site'][0], P1['site'][0] * slope + intercept],
            [P2['site'][0], P2['site'][0] * slope + intercept],
        ]
        up = False

    bisector = {
        'sites': [P1, P2],
        'up': up,
        'points': [],
        'intersections': [],
        'compound': False,
    }

    if up:
        sortedVerts = sorted(vertexes, key=lambda a: a[1])
        bisector['points'] = sorted(
            [[sortedVerts[0][0], 0],
             sortedVerts[0],
             sortedVerts[1],
             [sortedVerts[1][0], height]],
            key=lambda a: a[1],
        )
    else:
        sortedVerts = sorted(vertexes, key=lambda a: a[0])
        bisector['points'] = sorted(
            [[0, sortedVerts[0][1]],
             sortedVerts[0],
             sortedVerts[1],
             [width, sortedVerts[1][1]]],
            key=lambda a: a[0],
        )

    bisector['points'] = _dedup_consecutive(bisector['points'])
    return bisector


# ---------------------------------------------------------------------------
# Bisector helpers
# ---------------------------------------------------------------------------

def curryFindBisector(callback, width, height):
    """Curry the findBisector function with width and height."""
    def findBisector(P1, P2):
        return callback(P1, P2, width, height)
    return findBisector


def findHopTo(bisector, hopFrom):
    """Return the site on the other side of a bisector."""
    return bisector['sites'][0] if bisector['sites'][1] is hopFrom else bisector['sites'][1]


def isBisectorTrapped(trapPoint, bisector):
    """Check if all points of a bisector are closer to trapPoint than to either of its own sites."""
    tp = trapPoint['site']
    s0 = bisector['sites'][0]['site']
    s1 = bisector['sites'][1]['site']
    for point in bisector['points']:
        dtp = distance(tp, point)
        if dtp > distance(s0, point) or dtp > distance(s1, point):
            return False
    return True


def getExtremePoint(bisector, goUp):
    """Get the highest (goUp=True) or lowest (goUp=False) y-coordinate of a bisector."""
    if goUp:
        return max(pt[1] for pt in bisector['points'])
    else:
        return min(pt[1] for pt in bisector['points'])


def clearOutOrphans(orphanage, trapPoint):
    """Remove bisectors trapped by trapPoint from orphanage's bisector list."""
    return [b for b in orphanage['bisectors'] if not isBisectorTrapped(trapPoint, b)]


# ---------------------------------------------------------------------------
# Bisector trimming and direction
# ---------------------------------------------------------------------------

def trimBisector(target, intersector, intersection):
    """Trim a bisector at an intersection point, discarding points inside the other polygon."""

    # Find the site from intersector that is NOT in target
    polygonSite = None
    for e in intersector['sites']:
        found = False
        for d in target['sites']:
            if d is e:
                found = True
                break
        if not found:
            polygonSite = e
            break

    ps = polygonSite['site']
    s0 = target['sites'][0]['site']
    s1 = target['sites'][1]['site']

    newPoints = [
        e for e in target['points']
        if distance(e, s0) < distance(e, ps)
        and distance(e, s1) < distance(e, ps)
    ]

    newPoints.append(intersection)

    if target['up']:
        newPoints.sort(key=lambda a: a[1])
    else:
        newPoints.sort(key=lambda a: a[0])

    target['points'] = newPoints


def isNewBisectorUpward(hopTo, hopFrom, site, goUp):
    """Determine if a new bisector is traveling upward relative to the merge line."""

    if hopTo['site'][0] - site['site'][0] == 0:
        # Vertical line: check if site is above hopTo
        return site['site'][1] > hopTo['site'][1]

    slope = (hopTo['site'][1] - site['site'][1]) / (hopTo['site'][0] - site['site'][0])
    intercept = hopTo['site'][1] - slope * hopTo['site'][0]

    isAboveLine = hopFrom['site'][1] > (slope * hopFrom['site'][0] + intercept)
    return isAboveLine


def determineFirstBorderCross(cropR, cropL, currentCropPoint):
    """Determine which border (left or right) intersection is closer vertically."""
    if abs(cropR['point'][1] - currentCropPoint[1]) == abs(cropL['point'][1] - currentCropPoint[1]):
        return None
    elif abs(cropR['point'][1] - currentCropPoint[1]) < abs(cropL['point'][1] - currentCropPoint[1]):
        return "right"
    else:
        return "left"


# ---------------------------------------------------------------------------
# Starting bisector and correctness checks
# ---------------------------------------------------------------------------

def findCorrectW(w, nearestNeighbor, findBisector):
    """Ensure the starting point w won't result in a trapped bisector."""

    startingBisector = findBisector(w, nearestNeighbor)

    wTrapList = []
    for e in w['bisectors']:
        hopTo = findHopTo(e, w)
        wTrapList.append({
            'hopTo': hopTo,
            'isTrapped': isBisectorTrapped(hopTo, startingBisector),
        })

    trapped = [x for x in wTrapList if x['isTrapped']]
    trapped.sort(key=lambda x: distance(x['hopTo']['site'], nearestNeighbor['site']))

    if trapped:
        return findCorrectW(trapped[0]['hopTo'], nearestNeighbor, findBisector)
    else:
        return w


def checkForOphans(trapper, trapped, goUp, findBisector):
    """Recursively check for orphaned bisectors."""

    candidates = []
    for bisector in trapped['bisectors']:
        hopTo = findHopTo(bisector, trapped)
        if goUp == (hopTo['site'][1] < trapped['site'][1]) and isBisectorTrapped(trapper, bisector):
            candidates.append(bisector)

    if not candidates:
        return None

    def sortKey(bisector):
        hopToA = findHopTo(bisector, trapped)
        mergeLineA = findBisector(hopToA, trapper)
        extremeA = getExtremePoint(mergeLineA, goUp)
        return -extremeA if goUp else extremeA

    candidates.sort(key=sortKey)
    return candidates[0]


def determineStartingBisector(w, nearestNeighbor, width, lastIntersect, findBisector):
    """Determine the starting bisector for the merge process."""

    z = [width, w['site'][1]]

    if lastIntersect is None:
        lastIntersect = w['site']

    zline = {'points': [w['site'], z]}

    intersection = None
    for bisector in nearestNeighbor['bisectors']:
        pt = bisectorIntersection(zline, bisector)
        if pt:
            intersection = {'point': pt, 'bisector': bisector}
            break

    if (intersection is not None
            and distance(w['site'], intersection['point'])
            > distance(nearestNeighbor['site'], intersection['point'])):
        startingBisector = findBisector(w, nearestNeighbor)
        return {
            'startingBisector': startingBisector,
            'w': w,
            'nearestNeighbor': nearestNeighbor,
            'startingIntersection': intersection['point'],
        }
    elif (intersection is not None
          and distance(w['site'], intersection['point'])
          < distance(nearestNeighbor['site'], intersection['point'])
          and intersection['point'][0] > lastIntersect[0]):
        nextR = findHopTo(intersection['bisector'], nearestNeighbor)
        return determineStartingBisector(w, nextR, width, intersection['point'], findBisector)
    else:
        w = findCorrectW(w, nearestNeighbor, findBisector)
        startingBisector = findBisector(w, nearestNeighbor)
        return {
            'startingBisector': startingBisector,
            'w': w,
            'nearestNeighbor': nearestNeighbor,
            'startingIntersection': intersection['point'] if intersection is not None else w['site'],
        }


# ---------------------------------------------------------------------------
# Merge line walking
# ---------------------------------------------------------------------------

def walkMergeLine(currentR, currentL, currentBisector, currentCropPoint, goUp,
                  crossedBorder, mergeArray, findBisector):
    """Walk along the merge line, trimming bisectors at intersections."""

    # Ensure currentBisector connects currentR and currentL
    if not all(e is currentR or e is currentL for e in currentBisector['sites']):
        currentBisector = findBisector(currentR, currentL)
        trimBisector(currentBisector, crossedBorder, currentCropPoint)
        mergeArray.append(currentBisector)

    # Find intersections with currentL's bisectors
    cropLArray = []
    for e in currentL['bisectors']:
        pt = bisectorIntersection(currentBisector, e)
        if not pt:
            continue
        hopTo = findHopTo(e, currentL)
        if (goUp == isNewBisectorUpward(hopTo, currentL, currentR, goUp)
                and (not samePoint(pt, currentCropPoint) or e is not crossedBorder)):
            cropLArray.append({'bisector': e, 'point': pt})

    cropLArray.sort(key=lambda item: angle(
        currentL['site'],
        findHopTo(item['bisector'], currentL)['site'],
    ), reverse=True)

    # Filter out invalid candidates
    filteredL = []
    for e in cropLArray:
        hopTo = findHopTo(e['bisector'], currentL)
        newMergeLine = findBisector(currentR, hopTo)
        trimBisector(newMergeLine, e['bisector'], e['point'])
        keep = True
        for d in cropLArray:
            dHopTo = findHopTo(d['bisector'], currentL)
            if isBisectorTrapped(dHopTo, newMergeLine) and dHopTo is not hopTo:
                keep = False
                break
        if keep:
            filteredL.append(e)
    cropLArray = filteredL

    # Find intersections with currentR's bisectors
    cropRArray = []
    for e in currentR['bisectors']:
        pt = bisectorIntersection(currentBisector, e)
        if not pt:
            continue
        hopTo = findHopTo(e, currentR)
        if (goUp == isNewBisectorUpward(hopTo, currentR, currentL, goUp)
                and (not samePoint(pt, currentCropPoint) or e is not crossedBorder)):
            cropRArray.append({'bisector': e, 'point': pt})

    cropRArray.sort(key=lambda item: angle(
        currentR['site'],
        findHopTo(item['bisector'], currentR)['site'],
    ))

    # Filter out invalid candidates
    filteredR = []
    for e in cropRArray:
        hopTo = findHopTo(e['bisector'], currentR)
        newMergeLine = findBisector(currentL, hopTo)
        trimBisector(newMergeLine, e['bisector'], e['point'])
        keep = True
        for d in cropRArray:
            dHopTo = findHopTo(d['bisector'], currentR)
            if isBisectorTrapped(dHopTo, newMergeLine) and dHopTo is not hopTo:
                keep = False
                break
        if keep:
            filteredR.append(e)
    cropRArray = filteredR

    # Select the best candidate from each side
    if goUp:
        sentinel = [float('inf'), float('inf')]
    else:
        sentinel = [-float('inf'), -float('inf')]

    if (len(cropLArray) > 0
            and cropLArray[0]['bisector'] is not currentBisector):
        cropL = cropLArray[0]
    else:
        cropL = {'bisector': None, 'point': sentinel}

    if (len(cropRArray) > 0
            and cropRArray[0]['bisector'] is not currentBisector):
        cropR = cropRArray[0]
    else:
        cropR = {'bisector': None, 'point': sentinel}

    # If no intersections, check for orphans and return
    if cropL['bisector'] is None and cropR['bisector'] is None:
        leftOrphan = checkForOphans(currentR, currentL, goUp, findBisector)
        rightOrphan = checkForOphans(currentL, currentR, goUp, findBisector)

        if leftOrphan is not None:
            for site_obj in leftOrphan['sites']:
                site_obj['bisectors'] = [
                    b for b in site_obj['bisectors'] if b is not leftOrphan
                ]
            hopTo = findHopTo(leftOrphan, currentL)
            currentR = findCorrectW(currentR, hopTo, findBisector)
            newMergeBisector = findBisector(hopTo, currentR)
            mergeArray.append(newMergeBisector)
            return walkMergeLine(
                currentR, hopTo, newMergeBisector, currentCropPoint,
                goUp, crossedBorder, mergeArray, findBisector,
            )

        if rightOrphan is not None:
            for site_obj in rightOrphan['sites']:
                site_obj['bisectors'] = [
                    b for b in site_obj['bisectors'] if b is not rightOrphan
                ]
            hopTo = findHopTo(rightOrphan, currentR)
            currentL = findCorrectW(currentL, hopTo, findBisector)
            newMergeBisector = findBisector(hopTo, currentL)
            mergeArray.append(newMergeBisector)
            return walkMergeLine(
                hopTo, currentL, newMergeBisector, currentCropPoint,
                goUp, crossedBorder, mergeArray, findBisector,
            )

        return mergeArray

    # Determine which side to cross first and update state
    direction = determineFirstBorderCross(cropR, cropL, currentCropPoint)

    if direction == "right":
        trimBisector(cropR['bisector'], currentBisector, cropR['point'])
        trimBisector(currentBisector, cropR['bisector'], cropR['point'])
        currentBisector['intersections'].append(cropR['point'])
        crossedBorder = cropR['bisector']
        currentR = findHopTo(cropR['bisector'], currentR)
        currentCropPoint = cropR['point']
    elif direction == "left":
        trimBisector(cropL['bisector'], currentBisector, cropL['point'])
        trimBisector(currentBisector, cropL['bisector'], cropL['point'])
        currentBisector['intersections'].append(cropL['point'])
        crossedBorder = cropL['bisector']
        currentL = findHopTo(cropL['bisector'], currentL)
        currentCropPoint = cropL['point']
    else:
        # Both intersect at the same distance: handle both
        trimBisector(cropR['bisector'], currentBisector, cropR['point'])
        trimBisector(currentBisector, cropR['bisector'], cropR['point'])
        currentBisector['intersections'].append(cropR['point'])
        crossedBorder = cropR['bisector']
        currentR = findHopTo(cropR['bisector'], currentR)
        currentCropPoint = cropR['point']

        trimBisector(cropL['bisector'], currentBisector, cropL['point'])
        trimBisector(currentBisector, cropL['bisector'], cropL['point'])
        currentBisector['intersections'].append(cropL['point'])
        crossedBorder = cropL['bisector']
        currentL = findHopTo(cropL['bisector'], currentL)
        currentCropPoint = cropL['point']

    return walkMergeLine(
        currentR, currentL, currentBisector, currentCropPoint,
        goUp, crossedBorder, mergeArray, findBisector,
    )


# ---------------------------------------------------------------------------
# Divide-and-conquer merge
# ---------------------------------------------------------------------------

def recursiveSplit(splitArray, findBisector, width, height):
    """Recursively split and merge sets of points (divide-and-conquer)."""

    if len(splitArray) > 2:
        splitPoint = (len(splitArray) - len(splitArray) % 2) // 2

        L = recursiveSplit(splitArray[:splitPoint], findBisector, width, height)
        R = recursiveSplit(splitArray[splitPoint:], findBisector, width, height)

        R.sort(key=lambda a: distance(L[-1]['site'], a['site']))

        startingInfo = determineStartingBisector(
            L[-1], R[0], width, None, findBisector,
        )

        initialBisector = startingInfo['startingBisector']
        initialR = startingInfo['nearestNeighbor']
        initialL = startingInfo['w']

        upStrokeArray = walkMergeLine(
            initialR, initialL, initialBisector, [width, height],
            True, None, [], findBisector,
        )
        downStrokeArray = walkMergeLine(
            initialR, initialL, initialBisector, [0, 0],
            False, None, [], findBisector,
        )

        mergeArray = [initialBisector] + upStrokeArray + downStrokeArray

        for bisector in mergeArray:
            bisector['mergeLine'] = len(splitArray)
            bisector['sites'][0]['bisectors'] = clearOutOrphans(
                bisector['sites'][0], bisector['sites'][1],
            )
            bisector['sites'][1]['bisectors'] = clearOutOrphans(
                bisector['sites'][1], bisector['sites'][0],
            )
            for site_obj in bisector['sites']:
                site_obj['bisectors'].append(bisector)

        return L + R

    elif len(splitArray) == 2:
        bisector = findBisector(splitArray[0], splitArray[1])
        for e in splitArray:
            e['bisectors'].append(bisector)
        return splitArray

    else:
        return splitArray


# ---------------------------------------------------------------------------
# Preprocessing: data cleaning
# ---------------------------------------------------------------------------

def cleanData(data):
    """Nudge points that form a square (|dx| == |dy|) to avoid degenerate bisectors."""
    for i, e in enumerate(data):
        for j, d in enumerate(data):
            if i != j and abs(d[0] - e[0]) == abs(d[1] - e[1]):
                d[0] = d[0] + 1e-10 * d[1]
                d[1] = d[1] + 2e-10 * d[0]
    return data


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generateL1Voronoi(sitePoints, width, height, nudgeData=True):
    """Generate an L1 (Manhattan distance) Voronoi diagram.

    Args:
        sitePoints: list of [x, y] coordinates
        width: width of the bounding box
        height: height of the bounding box
        nudgeData: if True, nudge points to avoid degenerate bisectors

    Returns:
        List of site dicts with 'site', 'bisectors', 'polygonPoints', 'd', 'neighbors'
    """

    if nudgeData:
        sitePoints = cleanData(sitePoints)

    # Sort by x, break ties with y
    sitePoints.sort(key=lambda a: (a[0], a[1]))

    # Remove duplicates (not in original JS, but prevents crashes)
    unique = []
    for p in sitePoints:
        if not unique or not samePoint(unique[-1], p):
            unique.append(p)
    sitePoints = unique

    sites = [{'site': e, 'bisectors': []} for e in sitePoints]

    findBisector = curryFindBisector(findL1Bisector, width, height)
    graph = recursiveSplit(sites, findBisector, width, height)

    def isPointonEdge(point):
        return (point[0] == 0 or point[0] == width
                or point[1] == 0 or point[1] == height)

    def arePointsOnSameEdge(P1, P2):
        return ((P1[0] == P2[0] and P1[0] == 0)
                or (P1[0] == P2[0] and P1[0] == width)
                or (P1[1] == P2[1] and P1[1] == 0)
                or (P1[1] == P2[1] and P1[1] == height))

    corners = [[0, 0], [width, 0], [width, height], [0, height]]

    for site in graph:
        # Chain bisector points together to form polygon boundary
        total_acc = None
        for index, bisector in enumerate(site['bisectors']):
            if index == 0:
                # Find a bisector on an edge if one exists
                startBisector = bisector
                for e in site['bisectors']:
                    if any(isPointonEdge(pt) for pt in e['points']):
                        startBisector = e
                        break

                startingPoints = list(startBisector['points'])
                if isPointonEdge(startingPoints[-1]):
                    startingPoints = startingPoints[::-1]

                total_acc = {'points': startingPoints, 'used': [startBisector]}
            else:
                last = total_acc['points'][-1]

                best_next = None
                best_dist = float('inf')
                for e in site['bisectors']:
                    if any(e is d for d in total_acc['used']):
                        continue
                    e0 = e['points'][0]
                    e1 = e['points'][-1]
                    d0 = distance(last, e0)
                    d1 = distance(last, e1)
                    eDist = d0 if d0 < d1 else d1
                    if eDist < best_dist:
                        best_dist = eDist
                        best_next = e

                nextPoints = list(best_next['points'])
                if samePoint(nextPoints[-1], last):
                    nextPoints = nextPoints[::-1]

                total_acc = {
                    'points': total_acc['points'] + nextPoints,
                    'used': total_acc['used'] + [best_next],
                }

        site['polygonPoints'] = total_acc['points'] if total_acc is not None else []
        site['polygonPoints'] = _dedup_consecutive(site['polygonPoints'])

        poly = site['polygonPoints']
        if (len(poly) >= 2
                and isPointonEdge(poly[0])
                and isPointonEdge(poly[-1])
                and not arePointsOnSameEdge(poly[0], poly[-1])):
            filteredCorners = [
                c for c in corners
                if all(not bisectorIntersection({'points': [c, site['site']]}, d)
                       for d in site['bisectors'])
            ]
            site['polygonPoints'] = poly + filteredCorners
            site['polygonPoints'] = _dedup_consecutive(site['polygonPoints'])

        site['polygonPoints'].sort(key=lambda a: angle(site['site'], a))
        site['polygonPoints'] = _dedup_consecutive(site['polygonPoints'])
        site['d'] = ('M '
                     + ' L'.join(f'{x} {y}' for x, y in site['polygonPoints'])
                     + ' Z')
        site['neighbors'] = [
            findHopTo(e, site)['site'] for e in site['bisectors']
        ]

    return graph


# ---------------------------------------------------------------------------
# Naive brute-force Voronoi (for reference / comparison)
# ---------------------------------------------------------------------------

def generateVoronoiPoints(points, width, height, distanceCallback):
    """Generate Voronoi pixel data using a naive brute-force approach."""
    import random
    colors = [
        {'point': e, 'color': [int(random.random() * 255) for _ in range(3)]}
        for e in points
    ]

    imageData = []
    for index in range(width * height):
        coordinate = [index % width, index // width]

        closest = {'point': [float('inf'), float('inf')]}
        for col in colors:
            closest = _reduce_closest(closest, col, coordinate, distanceCallback)

        if isinstance(closest, list):
            imageData.append([0, 0, 0])
        else:
            imageData.append(closest['color'])

    return imageData


def _reduce_closest(c, e, coordinate, distanceCallback):
    if isinstance(c, list):
        if all(distanceCallback(d['point'], coordinate) < distanceCallback(e['point'], coordinate)
               for d in c):
            return c
        return e
    elif distanceCallback(c['point'], coordinate) == distanceCallback(e['point'], coordinate):
        return [c, e]
    else:
        if distanceCallback(c['point'], coordinate) < distanceCallback(e['point'], coordinate):
            return c
        return e


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    'generateVoronoiPoints',
    'generateL1Voronoi',
    'cleanData',
    'findL1Bisector',
    'bisectorIntersection',
    'distance',
    'samePoint',
]
