import hashlib
import json
import sys
from collections import Counter

from app.preprocess import preprocess
from own import OWN_DIR, is_test, rows

info = json.loads(open(sys.argv[1]).read())
classes, errors, seen, names = info["classes"], [], {}, set()
records = rows()

for r in records:
    if r["file"] in names:
        errors.append(f"{r['file']}: listed twice in labels.csv")
        continue
    names.add(r["file"])
    f = OWN_DIR / r["file"]
    if not f.exists():
        errors.append(f"{r['file']}: file missing")
        continue
    png = f.read_bytes()
    digest = hashlib.md5(png).hexdigest()
    if digest in seen:
        errors.append(f"{r['file']}: duplicate of {seen[digest]}")
    seen[digest] = r["file"]
    if r["label"] not in classes:
        errors.append(f"{r['file']}: unknown label {r['label']!r}")
    try:
        if preprocess(png, info["dataset"]) is None:
            errors.append(f"{r['file']}: blank drawing")
    except Exception as e:
        errors.append(f"{r['file']}: unreadable ({e})")

listed = {r["file"] for r in records} | {"labels.csv"}
errors += [f"{f.name}: not in labels.csv" for f in OWN_DIR.glob("*") if f.name not in listed]

counts = Counter(r["label"] for r in records)
test = sum(is_test(r["file"]) for r in records)
print(f"{len(records)} drawings ({len(records) - test} train / {test} own-test), {len(counts)}/{len(classes)} classes")
print("  ".join(f"{c}:{counts[c]}" for c in classes))
missing = [c for c in classes if counts[c] == 0]
if missing:
    print(f"no drawings yet for: {' '.join(missing)}")

if errors:
    print(f"\n{len(errors)} problem(s):")
    print("\n".join(f"  {e}" for e in errors))
    sys.exit(1)
print("data ok")
