#!/usr/bin/env bash
# New data -> train on Colab -> gate -> image -> deploy. FORCE=1 skips the retrain check.
set -euo pipefail

STORE=${STORE:-$HOME/.local/share/digit-classifier}
PY="uv run --no-project --with mlflow==3.16.1 --with-requirements app/requirements.txt python"
step() { printf '\n==> %s\n' "$*"; }

step "0. clean working tree"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "commit your changes first, so the model records which code trained it"
  exit 1
fi

step "1. should we retrain?"
reason=$($PY trigger.py "$STORE")
if [ -z "$reason" ] && [ "${FORCE:-0}" != 1 ]; then
  echo "nothing to do"
  exit 0
fi
echo "yes: ${reason:-forced}"

step "2. validate and version data"
make data
if ! git diff --quiet collected.dvc .gitignore; then
  git add collected.dvc .gitignore
  git commit -qm "data: update own drawings"
  echo "committed new data version"
fi

COMMIT=$(git rev-parse --short HEAD)
step "3. train on Colab (code $COMMIT)"
trap 'cloudmake --stop >/dev/null 2>&1 || true' EXIT
cloudmake --collect out results DATASET=emnist MODEL=resnet SCHED=cosine OWN=1 COMMIT="$COMMIT"
cloudmake --stop >/dev/null 2>&1 || true
make track

step "4. promote"
if ! make promote; then
  echo "not deployed: see the reason above; production stays as it was"
  exit 0
fi

step "5. bundle and build image"
make bundle
VERSION=$(python3 -c "import json; print(json.load(open('app/model/info.json'))['version'])")
IMAGE="digit-app:v$VERSION-$COMMIT"
docker build -q -t "$IMAGE" . >/dev/null
echo "built $IMAGE"

step "6. deploy with health check"
./deploy.sh "$IMAGE" "$VERSION"
