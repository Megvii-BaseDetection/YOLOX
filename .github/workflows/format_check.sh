#!/bin/bash -e

set -e

export PYTHONPATH=$PWD:$PYTHONPATH

flake8 yolox || flake8_ret=$?
if [ "$flake8_ret" ]; then
    exit $flake8_ret
fi
echo "All flake check passed!"
isort --check-only -rc yolox || isort_ret=$?
if [ "$isort_ret" ]; then
    exit $isort_ret
fi
echo "All isort check passed!"
