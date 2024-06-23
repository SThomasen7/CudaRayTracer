#!/bin/bash

./clean.sh >> /dev/null
make
./test_runner

echo "ALL TESTS TERMINATED"
echo "$(find -name *.success|wc -l) success(es)"
echo "$(find -name *.fail|wc -l)    fail(s)"
