import cv2

from .package_finder import iou

FONT = cv2.FONT_HERSHEY_SIMPLEX
CALM = (220, 220, 220)
BUSY = (0, 165, 255)
GREEN = (60, 220, 60)
PERSON = (60, 60, 230)


def label(img, x, y, text):
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
    x = max(x, 0)
    y = max(y, th + 8)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 8, y), GREEN, -1)
    cv2.putText(img, text, (x + 4, y - 5), FONT, 0.5, (0, 0, 0), 1)


def draw_box(img, box, color, thickness):
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)


def best_shelf_match(shelf, box):
    best_id = None
    best_iou = 0.0
    for shelf_box, pkg_id in shelf:
        score = iou(shelf_box, box)
        if score > best_iou:
            best_id = pkg_id
            best_iou = score
    if best_iou > 0.3:
        return best_id
    return None


def draw_overlay(frame, watchers, person_boxes, items=(), min_score=0.6):
    img = frame.copy()
    shelf = []
    for watcher in watchers:
        z = watcher.zone
        if watcher.busy:
            color = BUSY
        else:
            color = CALM
        draw_box(img, (z.x, z.y, z.w, z.h), color, 1)
        cv2.putText(img, z.name, (z.x + 4, z.y + 18), FONT, 0.55, color, 2)
        for pkg_id, view in watcher.views.items():
            x, y, w, h = view.bbox
            shelf.append(((z.x + x, z.y + y, w, h), pkg_id))

    drawn = set()
    for item in items:
        x, y, w, h = item.box
        if not item.still:
            draw_box(img, item.box, GREEN, 1)
            label(img, x, y, "bevaeger sig")
            continue
        if item.score < min_score:
            continue
        text = f"pakke {item.score:.0%}"
        pkg_id = best_shelf_match(shelf, item.box)
        if pkg_id is not None:
            text = f"#{pkg_id} {text}"
            drawn.add(pkg_id)
        draw_box(img, item.box, GREEN, 3)
        label(img, x, y, text)

    for box, pkg_id in shelf:
        if pkg_id not in drawn:
            x, y, w, h = box
            draw_box(img, box, GREEN, 3)
            label(img, x, y, f"#{pkg_id}")
    for box in person_boxes:
        draw_box(img, box, PERSON, 2)
    return img
