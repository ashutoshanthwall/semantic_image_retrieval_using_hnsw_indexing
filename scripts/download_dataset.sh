#!/usr/bin/env bash
# Downloads Flickr30k (images + 5 captions each) from Hugging Face nlphuji/flickr30k and
# verifies the files against the SHA-256 recorded by the Hugging Face LFS metadata.
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=data/raw/flickr30k
BASE=https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main
mkdir -p "$RAW"
[ -f "$RAW/flickr_annotations_30k.csv" ] || curl -fL -o "$RAW/flickr_annotations_30k.csv" "$BASE/flickr_annotations_30k.csv"
[ -f "$RAW/flickr30k-images.zip" ] || curl -fL -C - -o "$RAW/flickr30k-images.zip" "$BASE/flickr30k-images.zip"
sha256sum -c - <<SUMS
2ce2420c0d17f0531deaa89ac657b4d5067ec519da16ff1ea12acbf5619c7391  $RAW/flickr30k-images.zip
395990db603ab8bafd5c7ab2746b22058bb1e75b78b3eb56ad755931364ac137  $RAW/flickr_annotations_30k.csv
SUMS
