import base64
import io
import json
import random
import sys
import urllib.request

from PIL import Image, ImageFilter

from training.own import OWN_DIR, rows

style, n, url = sys.argv[1], int(sys.argv[2]), sys.argv[3]
random.seed(0)


def change(img):
    if style == "thick":
        return img.filter(ImageFilter.MaxFilter(21))
    if style == "rotate":
        return img.rotate(35)
    if style == "noise":
        px = img.load()
        for _ in range(60):
            x, y = random.randrange(275), random.randrange(275)
            for dx in range(5):
                for dy in range(5):
                    px[x + dx, y + dy] = 255
    return img


files = [r["file"] for r in rows()]
for _ in range(n):
    img = change(Image.open(OWN_DIR / random.choice(files)).convert("L"))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    body = json.dumps({"image": base64.b64encode(buf.getvalue()).decode()}).encode()
    urllib.request.urlopen(urllib.request.Request(f"{url}/predict", body, {"Content-Type": "application/json"}))
print(f"sent {n} {style} drawings to {url}")
