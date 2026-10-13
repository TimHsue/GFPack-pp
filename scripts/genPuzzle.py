import argparse
import copy
import random
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, LineString
from shapely.geometry import Polygon
from tqdm import trange
from shapely.affinity import rotate
import pickle
import shapely

class TPoint:

    def __init__(self, x, y) -> None:
        self.x = float(x)
        self.y = float(y)

    def __add__(self, other):
        return TPoint(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return TPoint(self.x - other.x, self.y - other.y)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return TPoint(self.x / other, self.y / other)
        raise TypeError("TPoint can only be divided by a scalar")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return TPoint(self.x * other, self.y * other)
        else:
            return self.x * other.x + self.y * other.y

    def __xor__(self, other):
        return self.x * other.y - self.y * other.x

    def __lt__(self, other):
        return self.x < other.x or (self.x == other.x and self.y < other.y)

    def __str__(self):
        return f"({self.x}, {self.y})"

    def mod(self):
        return (self.x**2 + self.y**2)**0.5

    def unit(self):
        length = self.mod()
        if length == 0:
            raise ValueError("Cannot normalize a zero-length direction")
        return self / length

    def normal(self):
        return TPoint(self.y, -self.x)


def inPoly(point, poly):
    pointList = [[p.x, p.y] for p in poly]
    spPoly = Polygon(pointList)
    spPoint = Point(point.x, point.y)
    return spPoly.contains(spPoint)

def intersection(lineA, lineB):
    sgLineA = LineString([Point(lineA[0].x, lineA[0].y), Point(lineA[1].x, lineA[1].y)])
    sgLineB = LineString([Point(lineB[0].x, lineB[0].y), Point(lineB[1].x, lineB[1].y)])

    if sgLineA.intersects(sgLineB) == False:
        return None
    intersectionPoint = sgLineA.intersection(sgLineB)
    if type(intersectionPoint) == Point:
        return TPoint(intersectionPoint.x, intersectionPoint.y)
    else:
        return None


def choseInit(poly):
    segmentLen = []
    for i in range(len(poly)):
        segmentLen.append((poly[i] - poly[(i + 1) % len(poly)]).mod())
    weight = np.array(segmentLen)
    segmentId = np.random.choice(len(poly), p=weight / weight.sum())
    pointA = poly[segmentId]
    pointB = poly[(segmentId + 1) % len(poly)]
    midPoint = pointA + (pointB - pointA) * (random.random() * 0.6 + 0.2)
    initDir = (pointB - pointA).unit().normal()
    # normal() points right: it is outward for a counterclockwise ring.
    if Polygon([(p.x, p.y) for p in poly]).exterior.is_ccw:
        initDir = initDir * -1.0
    initDir = (initDir + TPoint((random.random() - 0.5) * 0.9, (random.random() - 0.5) * 0.9)).unit()
    return midPoint, initDir, segmentId

def samePoint(pointA, pointB):
    return abs(pointA.x - pointB.x) < 1e-2 and abs(pointA.y - pointB.y) < 1e-2

def minCrossPoint(lineA, poly, ignFirst):
    crossList, iList = [], []
    for i in range(len(poly)):
        lineB = [poly[i], poly[(i + 1) % len(poly)]]
        cross = intersection(lineA, lineB)
        if cross != None:
            if ignFirst and samePoint(lineA[0], cross):
                continue
            crossList.append(cross)
            iList.append(i)

    if len(crossList) == 0:
        return None, None
    minT = 1e9
    for i in range(len(crossList)):
        t = (crossList[i] - lineA[0]) * (lineA[1] - lineA[0]) / ((lineA[1] - lineA[0]).mod() ** 2)
        if t < minT:
            minT = t
            minId = i
    return crossList[minId], iList[minId]

def deletePoint(poly):
    uniquePoly = []
    for point in poly:
        if len(uniquePoly) == 0 or uniquePoly[-1] != [point.x, point.y]:
            uniquePoly.append([point.x, point.y])
    polygon = Polygon(uniquePoly)
    # polygon = Polygon([[p.x, p.y] for p in poly])
    simplifiedPolygon = polygon.simplify(0.000001, preserve_topology=True)
    newPolygon = []
    for point in simplifiedPolygon.exterior.coords:
        newPolygon.append(TPoint(point[0], point[1]))
    return newPolygon[:-1]

def crossPath(path, nowLine):
    for i in range(len(path) - 2):
        line = [path[i], path[i + 1]]
        cross = intersection(line, nowLine)
        if cross is not None:
            return True
    return False

def cutCurrent(poly, step=500.0, randomDeg=0.5, max_path_steps=2048):
    initPoint, initDir, initId = choseInit(poly)
    path = [copy.deepcopy(initPoint)]
    nowPoint = initPoint
    nowDir = initDir
    for _ in range(max_path_steps):
        nextPoint = nowPoint + nowDir * step
        nowLine = [nowPoint, nextPoint]

        loopTime = 0

        while crossPath(path, nowLine):
            nowDir = (nowDir + TPoint((random.random() - 0.5) * randomDeg, (random.random() - 0.5) * randomDeg)).unit()
            nextPoint = nowPoint + nowDir * step
            nowLine = [nowPoint, nextPoint]
            loopTime += 1
            if loopTime > 16:
                return None, None

        ignFirst = False
        if len(path) < 2:
            ignFirst = True
        newPoint, newId = minCrossPoint([nowPoint, nextPoint], poly, ignFirst)
        if newPoint == None:
            path.append(copy.deepcopy(nextPoint))
        else:
            path.append(copy.deepcopy(newPoint))

            splitPolyA = [poly[initId]]
            for point in path:
                splitPolyA.append(TPoint(point.x, point.y))
            nowIdInA = (newId + 1) % len(poly)
            while nowIdInA != initId:
                splitPolyA.append(poly[nowIdInA])
                nowIdInA = (nowIdInA + 1) % len(poly)

            splitPolyB = [poly[newId]]
            for point in path[::-1]:
                splitPolyB.append(TPoint(point.x, point.y))
            nowIdInB = (initId + 1) % len(poly)
            while nowIdInB != newId:
                splitPolyB.append(poly[nowIdInB])
                nowIdInB = (nowIdInB + 1) % len(poly)
            for splitPoly in (splitPolyA, splitPolyB):
                shape = Polygon([(p.x, p.y) for p in splitPoly])
                if not shape.is_valid or shape.area <= 0:
                    return None, None
            return deletePoint(splitPolyA), deletePoint(splitPolyB)
            # return splitPolyA, splitPolyB

        nowPoint = nextPoint
        nowDir = (nowDir + TPoint((random.random() - 0.5) * randomDeg, (random.random() - 0.5) * randomDeg)).unit()
    return None, None

def drawPoly(poly):
    pointList = [[p.x, p.y] for p in poly]
    pointList.append(pointList[0])
    plt.plot(*zip(*pointList), color='r')

def getBox(poly):
    minX, minY = 1e9, 1e9
    maxX, maxY = -1e9, -1e9
    if type(poly[0]) is TPoint:
        for point in poly:
            minX = min(minX, point.x)
            minY = min(minY, point.y)
            maxX = max(maxX, point.x)
            maxY = max(maxY, point.y)
    if type(poly[0]) is list:
        for point in poly:
            minX = min(minX, point[0])
            minY = min(minY, point[1])
            maxX = max(maxX, point[0])
            maxY = max(maxY, point[1])
    return minX, minY, maxX, maxY

def getXYTheta(poly, rotation):
    polyS = Polygon([[p.x, p.y] for p in poly])
    cetroid = polyS.centroid
    centerPoint = [cetroid.x, cetroid.y]

    if rotation:
        theta = random.random() * 2 * np.pi
    else:
        theta = 0
    rotatedPolyS = rotate(polyS, -theta, origin=centerPoint, use_radians=True)
    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)

    rotatedPoly = []
    for point in rotatedPolyS.exterior.coords:
        rotatedPoly.append(TPoint(point[0], point[1]))
    for point in rotatedPoly:
        point.x -= centerPoint[0]
        point.y -= centerPoint[1]
    if rotation == False:
        return [float(centerPoint[0]), float(centerPoint[1])], rotatedPoly
    else:
        return [float(centerPoint[0]), float(centerPoint[1]), float(cosTheta), float(sinTheta)], rotatedPoly


def genDataPoly(polys, cut=16, rotation=True, resample_num=64, step=500.0,
                max_cut_attempts=256, max_path_steps=2048):

    if not isinstance(cut, int) or cut < 0:
        raise ValueError("cut must be a non-negative integer")
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step must be positive and finite")
    if max_cut_attempts < 1 or max_path_steps < 1:
        raise ValueError("retry limits must be positive")
    if resample_num < 0 or 0 < resample_num < 3:
        raise ValueError("resample_num must be 0 or at least 3")
    if len(polys) != 1:
        raise ValueError("Expected one initial boundary polygon")
    mainPolygon = Polygon([(p.x, p.y) for p in polys[0]])
    if not mainPolygon.is_valid or mainPolygon.area <= 0:
        raise ValueError("Boundary must be a valid polygon with positive area")

    mainShape = copy.deepcopy(polys[0])
    cutTime = 0
    attempts = 0

    while cutTime < cut:
        attempts += 1
        if attempts > max_cut_attempts:
            raise RuntimeError(
                f"Could not complete cut {cutTime + 1}/{cut} after "
                f"{max_cut_attempts} attempts; try another seed, a larger step, "
                "or a simpler boundary"
            )
        areas = [Polygon([[p.x, p.y] for p in poly]).area for poly in polys]
        weight = np.array(areas)
        choosedId = np.argmax(weight)
        choosedPoly = polys[choosedId]
        areaBe = Polygon([[p.x, p.y] for p in choosedPoly]).area

        polyA, polyB = cutCurrent(choosedPoly, step, max_path_steps=max_path_steps)
        if polyA == None:
            continue
        shapeA = Polygon([(p.x, p.y) for p in polyA])
        shapeB = Polygon([(p.x, p.y) for p in polyB])
        source = Polygon([(p.x, p.y) for p in choosedPoly])
        tolerance = max(1e-8, areaBe * 1e-8)
        if (not shapeA.is_valid or not shapeB.is_valid
                or min(shapeA.area, shapeB.area) <= tolerance
                or shapeA.intersection(shapeB).area > tolerance
                or shapeA.union(shapeB).symmetric_difference(source).area > tolerance):
            continue
        polys.pop(choosedId)
        polys.append(polyA)
        polys.append(polyB)
        cutTime += 1
        attempts = 0


    inputPolys = []
    polyVertices = []
    polyTargets = []
    totalArea = 0
    for poly in polys:
        polyTarget, inputPoly = getXYTheta(poly, rotation)

        polyArea = Polygon([[p.x, p.y] for p in poly]).area
        totalArea += polyArea

        mainContour = []
        for point in inputPoly:
            mainContour.append([point.x, point.y])
        # * 1.05 to prevent the polygon from being out of the grid
        if resample_num > 0:
            n_samples = resample_num
            polyP = shapely.geometry.Polygon(mainContour)
            contour = polyP.exterior
            interval = np.linspace(0, 1, n_samples + 1)[:-1]
            pointsSH = [contour.interpolate(interval[i], normalized=True) for i in range(len(interval))]
            points = [point.coords[0] for point in pointsSH]
        else:
            points = mainContour
        # drawGrid(outArray.tolist())
        inputPolys.append(points)
        polyVertices.append(mainContour)
        polyTargets.append(polyTarget)

    mainShapeArea = Polygon([[p.x, p.y] for p in mainShape]).area
    if abs(totalArea - mainShapeArea) > 1:
        print("totalArea error", totalArea, mainShapeArea)
        return None, None, None

    return polyVertices, polyTargets, inputPolys

def calCenter(poly):
    avgX, avgY = 0, 0
    for point in poly:
        avgX += point[0]
        avgY += point[1]
    avgX /= len(poly)
    avgY /= len(poly)
    return avgX, avgY

def drawData(inputPolys, polyTargets, fileNmae="test.png"):
    for j, poly in enumerate(inputPolys):
        polyS = Polygon(poly)
        cetroid = polyS.centroid
        centerPoint = [cetroid.x, cetroid.y]
        # centerPoint = calCenter(poly)

        cosTheta = polyTargets[j][2] if len(polyTargets[j]) == 4 else 1.0
        sinTheta = polyTargets[j][3] if len(polyTargets[j]) == 4 else 0.0
        theta = np.arctan2(sinTheta, cosTheta)
        tx = polyTargets[j][0]
        ty = polyTargets[j][1]
        rotatedPolyS = rotate(polyS, theta, origin=(0, 0), use_radians=True)

        targetPoly = []
        for point in rotatedPolyS.exterior.coords:
            targetPoly.append(TPoint(point[0] + tx, point[1] + ty))
        drawPoly(targetPoly)
    plt.gca().set_aspect('equal')
    plt.savefig(fileNmae)
    plt.clf()

def scalePoly(poly, lenx, leny):
    if lenx > leny:
        scale = 2000.0 / lenx
    else:
        scale = 2000.0 / leny
    newPoly = []
    minX = 1e9
    minY = 1e9
    for point in poly:
        newPoly.append(TPoint(point[0] * scale, point[1] * scale))
        minX = min(minX, point[0] * scale)
        minY = min(minY, point[1] * scale)
    for point in newPoly:
        point.x -= minX
        point.y -= minY
    return newPoly


def scaleTo(poly, size):
    points = np.asarray(poly, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
            or not np.isfinite(points).all()):
        raise ValueError("Each boundary must be a finite N x 2 vertex array (N >= 3)")
    shape = Polygon(points)
    if not shape.is_valid or shape.area <= 0:
        raise ValueError("Boundary must be simple and have positive area")
    # Remove an optional repeated endpoint and normalize the winding.
    poly = list(shape.exterior.coords)[:-1]
    if not shape.exterior.is_ccw:
        poly.reverse()
    minX, minY = 1e9, 1e9
    maxX, maxY = -1e9, -1e9
    for point in poly:
        minX = min(minX, point[0])
        minY = min(minY, point[1])
        maxX = max(maxX, point[0])
        maxY = max(maxY, point[1])

    scale = size / max(maxX - minX, maxY - minY)
    newPoly = []
    for point in poly:
        newPoly.append(TPoint((point[0] - minX) * scale, (point[1] - minY) * scale))
    return newPoly

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cut', type=int, default=15, help="number of cuts; produces cut + 1 pieces")
    parser.add_argument('--rotation', type=int, default=1, choices=[0, 1], help="whether to rotate the polygon after cutting")
    parser.add_argument('--resample_num', type=int, default=0, help="number of points to resample, 0 means no resample")
    parser.add_argument('--datasize', type=int, default=16, help="number of data to generate")
    parser.add_argument('--rect', type=int, default=0, choices=[0, 1], help="whether to use rectangle boundary")
    parser.add_argument('--step', type=float, default=500.0, help="step length of cutting")
    parser.add_argument('--boundary', type=Path, default=Path('boundary.pkl'),
                        help="JSON or trusted pickle containing a list of boundary vertex arrays")
    parser.add_argument('--output', type=Path, default=Path('dataset.pkl'))
    parser.add_argument('--vis_dir', type=Path, default=Path('puzzle_vis'))
    parser.add_argument('--vis_count', type=int, default=16, help="0 disables visualization")
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--max_cut_attempts', type=int, default=256)
    parser.add_argument('--max_path_steps', type=int, default=2048)
    args = parser.parse_args(argv)
    if args.cut < 0 or args.datasize < 1 or args.vis_count < 0:
        parser.error('cut and vis_count must be non-negative; datasize must be positive')
    if not np.isfinite(args.step) or args.step <= 0:
        parser.error('step must be positive and finite')
    if args.max_cut_attempts < 1 or args.max_path_steps < 1:
        parser.error('retry limits must be positive')
    if args.resample_num < 0 or 0 < args.resample_num < 3:
        parser.error('resample_num must be 0 or at least 3')
    random.seed(args.seed)
    np.random.seed(args.seed)

    boundaries = []
    if not args.rect:
        try:
            with args.boundary.open('rb') as file:
                raw_boundaries = (json.load(file) if args.boundary.suffix.lower() == '.json'
                                  else pickle.load(file))
            if len(raw_boundaries) == 0:
                raise ValueError('Boundary list is empty')
            boundaries = [scaleTo(boundary, 2000) for boundary in raw_boundaries]
        except (OSError, ValueError, TypeError, EOFError, pickle.UnpicklingError) as exc:
            parser.error(f'Cannot load boundaries from {args.boundary}: {exc}')

    polyVerticesData = []
    polyTargetsData = []
    boundIdsData = []
    for i in trange(args.datasize):
        if not args.rect:
            choosedBoundId = random.randint(0, len(boundaries) - 1)
            boundary = boundaries[choosedBoundId]
        else:
            choosedBoundId = -1  # sentinel for the built-in rectangle
            size = 2000
            boundary = [TPoint(0, 0), TPoint(size, 0), TPoint(size, size), TPoint(0, size)]

        polys = []
        polys.append(copy.deepcopy(boundary))
        try:
            polyVertices, polyTargets, inputPolys = genDataPoly(
                polys, cut=args.cut, rotation=bool(args.rotation),
                resample_num=args.resample_num, step=args.step,
                max_cut_attempts=args.max_cut_attempts, max_path_steps=args.max_path_steps)
        except (ValueError, RuntimeError) as exc:
            parser.exit(1, f'Failed to generate sample {i}: {exc}\n')
        else:
            if polyVertices == None:
                parser.exit(1, f'Failed to generate sample {i}: area validation failed\n')
            polyVerticesData.append(polyVertices)
            polyTargetsData.append(polyTargets)
            boundIdsData.append(choosedBoundId)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as file:
        pickle.dump(polyVerticesData, file)
        pickle.dump(polyTargetsData, file)
        pickle.dump(boundIdsData, file)
    print(f'Generated {len(polyVerticesData)} samples in {args.output}')
    if args.vis_count:
        args.vis_dir.mkdir(parents=True, exist_ok=True)
        for i in range(min(args.vis_count, len(polyVerticesData))):
            drawData(polyVerticesData[i], polyTargetsData[i], str(args.vis_dir / f'vis{i}.png'))


if __name__ == "__main__":
    main()
