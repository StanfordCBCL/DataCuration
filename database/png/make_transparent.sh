#!/bin/bash
for f in original/*.png; do
	convert $f -transparent white $(basename "$f")
done
