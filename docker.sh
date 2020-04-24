#!/bin/bash
if [ -z $(docker images -q agad) ]; then
  echo "Image does not exist. Creating image agad ..."
  docker build -t agad .
fi
docker run -it --rm -w /agad agad
