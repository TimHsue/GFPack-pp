import random
import torch as tr


class Line:
    def __init__(self, pa, pb) -> None:
        self.beginPoint = pa
        self.endPoint = pb

        direction = self.endPoint - self.beginPoint
        self.dir = direction.unit()
        self.maxT = direction.mod()

    def getPointY(self, y):
        min_y = min(self.beginPoint.y, self.endPoint.y)
        max_y = max(self.beginPoint.y, self.endPoint.y)
        if y < min_y or y > max_y:
            return None
        if self.dir.y == 0:
            return Point(self.beginPoint.x, y)
        x = self.beginPoint.x + (y - self.beginPoint.y) * self.dir.x / self.dir.y
        return Point(x, y)


class Point:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __truediv__(self, other):
        if isinstance(other, (float, int)):
            return Point(self.x / other, self.y / other)
        raise TypeError("Point can only be divided by a scalar.")

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return Point(self.x * other, self.y * other)
        return self.x * other.x + self.y * other.y

    def mod(self):
        return (self.x**2 + self.y**2) ** 0.5

    def unit(self):
        length = self.mod()
        if length == 0:
            return Point(0.0, 0.0)
        return self / length


class Grid:
    def __init__(self, n, m, x, y) -> None:
        self.n = n
        self.m = m
        self.x = x
        self.y = y
        self.grid = [[0.0 for _ in range(m)] for __ in range(n)]

    def fillGrid(self, bx, by, tx, ty):
        for i in range(bx, tx + 1):
            for j in range(by, ty + 1):
                self.grid[i][j] = 1


class SweepLine:
    def __init__(self, points) -> None:
        self.points = points

    def solve(self, n, m, x, y, oldCrossPoints):
        dX = x / n
        dY = y / m
        crossPoints = []

        max_y = 0
        for point in self.points:
            max_y = max(max_y, point.y)

        nowY = 0.0
        nowRow = 0
        while (nowY + dY) <= max_y:
            crossPoints.append(oldCrossPoints[nowRow])
            for i in range(len(self.points)):
                ne = (i + 1) % len(self.points)
                line = Line(self.points[i], self.points[ne])
                if line.dir.y == 0:
                    continue
                crossPoint = line.getPointY(nowY)
                if crossPoint is None:
                    continue
                high_y = max(self.points[i].y, self.points[ne].y)
                if crossPoint.y == high_y:
                    continue
                crossPoint.x = int(crossPoint.x / dX + 0.5)
                crossPoint.y = nowRow
                crossPoints[nowRow].append(crossPoint)
            nowY += dY
            nowRow += 1
        while nowRow < m:
            crossPoints.append(oldCrossPoints[nowRow])
            nowRow += 1
        return crossPoints


class Polygon:
    def __init__(self, fileName, app=True) -> None:
        if app:
            self.contour, self.x, self.y = self.readPolygonApp(fileName)
        else:
            self.contour, self.x, self.y = self.readPolygon(fileName)
        maxPointCnt = 0
        maxId = 0
        for idx, points in enumerate(self.contour):
            if len(points) > maxPointCnt:
                maxPointCnt = len(points)
                maxId = idx
        self.mainContour = self.contour[maxId]
        self.grid = []
        self.maxSize = max(self.x, self.y)

    def readPolygon(self, fileName):
        contour = []
        min_x = float("inf")
        max_x = float("-inf")
        min_y = float("inf")
        max_y = float("-inf")
        with open(fileName, "r") as f:
            lines = f.readlines()
        block = int(lines[0])
        nowLine = 1
        for _ in range(block):
            points = []
            pointCnt = int(lines[nowLine])
            for _ in range(pointCnt):
                nowLine += 1
                x_str, y_str = lines[nowLine].split()
                x = float(x_str)
                y = float(y_str)
                points.append(Point(x, y))
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
            contour.append(points)
            nowLine += 1
        width = max(max_x - min_x, 0.0)
        height = max(max_y - min_y, 0.0)
        return contour, width, height

    def readPolygonApp(self, fileName):
        contour = []
        min_x = float("inf")
        max_x = float("-inf")
        min_y = float("inf")
        max_y = float("-inf")
        with open(fileName, "r") as f:
            lines = f.readlines()
        block = int(lines[0])
        nowLine = 1
        for _ in range(block):
            pointCnt = int(lines[nowLine])
            nowLine += 1
            points = []
            for _ in range(pointCnt):
                x_str, y_str = lines[nowLine].split()
                nowLine += 1
                x = float(x_str) * 10
                y = float(y_str) * 10
                points.append(Point(x, y))
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)
            contour.append(points)
            nowLine += 1
        width = max(max_x - min_x, 0.0)
        height = max(max_y - min_y, 0.0)
        return contour, width, height

    def getMaxContour(self):
        return [[point.x, point.y] for point in self.mainContour]

    def solve(self, n, m, x, y):
        self.grid = Grid(n, m, x, y)
        crossPoints = [[] for _ in range(m)]
        for contour in self.contour:
            if len(contour) <= 1:
                continue
            sweepLine = SweepLine(contour)
            crossPoints = sweepLine.solve(n, m, x, y, crossPoints)

        for row in crossPoints:
            row.sort(key=lambda point: point.x)
            for i in range(0, len(row), 2):
                self.grid.fillGrid(row[i].x, row[i].y, row[i + 1].x, row[i + 1].y + 1)
        return self.grid


def setRandomSeed(seed):
    random.seed(seed)
    tr.manual_seed(seed)
    tr.cuda.manual_seed(seed)
    tr.cuda.manual_seed_all(seed)
    tr.backends.cudnn.deterministic = True
