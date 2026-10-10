"""Controlled synthetic images for geometry checks, not a production benchmark."""
from pathlib import Path
import json

import cv2
import numpy as np


def scene(arcs=(), lines=(), dots=False, seed=17):
    rng = np.random.default_rng(seed)
    size = 1096
    yy, xx = np.mgrid[:size, :size]
    wafer = (xx - 548) ** 2 + (yy - 548) ** 2 <= 515 ** 2
    # Slight shading and sensor noise, with a visible circular wafer boundary.
    background = 40 + 3 * (xx - 548) / 515 + rng.normal(0, 1.3, (size, size))
    gray = np.where(wafer, background, 5).clip(0, 255).astype(np.uint8)
    for cx, cy, radius, start, span, intensity, width in arcs:
        cv2.ellipse(gray, (cx, cy), (radius, radius), 0, start, start + span,
                    intensity, width, cv2.LINE_AA)
    for x1, y1, x2, y2 in lines:
        cv2.line(gray, (x1, y1), (x2, y2), 150, 2, cv2.LINE_AA)
    if dots:
        for x, y in rng.integers(100, 996, size=(140, 2)):
            if wafer[y, x]:
                cv2.circle(gray, (int(x), int(y)), 1, 150, -1)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def cases():
    # Entries: name, image, expected scratch circles (cx,cy,r), expected rule hit.
    return [
        ("clean", scene(), [], False),
        ("full_eccentric", scene([(572, 530, 250, 0, 360, 130, 3)]),
         [(572, 530, 250)], True),
        ("arc35_eccentric", scene([(572, 530, 320, 70, 35, 130, 3)]),
         [(572, 530, 320)], True),
        ("wrap40", scene([(572, 530, 320, 350, 40, 130, 3)]),
         [(572, 530, 320)], True),
        ("weak45", scene([(566, 535, 280, 185, 45, 60, 2)]),
         [(566, 535, 280)], True),
        ("two_radii", scene([(572, 530, 150, 0, 360, 130, 2),
                              (572, 530, 300, 120, 80, 130, 2)]),
         [(572, 530, 150), (572, 530, 300)], True),
        ("arc20", scene([(572, 530, 320, 65, 20, 130, 2)]), [], False),
        ("lines", scene(lines=[(190, 360, 910, 720), (150, 570, 930, 570)]),
         [], False),
        ("dots", scene(dots=True), [], False),
        # Two distant 20-degree pieces must not be added to make 40 degrees.
        ("separate20", scene([(572, 530, 320, 40, 20, 130, 2),
                               (572, 530, 320, 190, 20, 130, 2)]), [], False),
    ]


def write_demo(destination):
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    truths = []
    for name, image, circles, hit in cases():
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("PNG encoding failed")
        encoded.tofile(str(target / f"{name}.png"))
        truths.append({"image": f"{name}.png", "expected_rule_hit": hit,
                       "circles": circles})
    (target / "truth.json").write_text(json.dumps(truths, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", nargs="?", default="synthetic_demo/input")
    write_demo(parser.parse_args().destination)
