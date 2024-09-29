#!/usr/bin/env bash

#bash ./build.sh

docker save eoir | gzip -c > eoir.tar.gz
