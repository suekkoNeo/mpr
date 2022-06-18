#!/bin/bash

cd mpr/ && python mpr.py && cd ../mcx && bin/mcx -f bin/test.json -F mc2 -O F -w D -G 1101 && sudo mv test.mc2 ../