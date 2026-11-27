import json


class Zone:
    def __init__(self, name, rect):
        self.name = name
        x = int(rect[0])
        y = int(rect[1])
        w = int(rect[2])
        h = int(rect[3])
        if x < 0:
            w = w + x
            x = 0
        if y < 0:
            h = h + y
            y = 0
        self.x = x
        self.y = y
        self.w = max(w, 0)
        self.h = max(h, 0)

    def crop(self, frame):
        return frame[self.y:self.y + self.h, self.x:self.x + self.w]

    def overlaps(self, box):
        bx, by, bw, bh = box
        if bx + bw < self.x or bx > self.x + self.w:
            return False
        if by + bh < self.y or by > self.y + self.h:
            return False
        return True


def load_zones(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for cam in data:
        zones = []
        for z in data[cam]:
            zones.append(Zone(z["name"], z["rect"]))
        result[cam] = zones
    return result
