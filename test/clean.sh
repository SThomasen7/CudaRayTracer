#!/bin/bash

# Clean up results
find -name *.err -exec rm {} \;
find -name *.out -exec rm {} \;
find -name *.success -exec rm {} \;
find -name *.fail -exec rm {} \;

make clean > /dev/null
