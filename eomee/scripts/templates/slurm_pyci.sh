#!/bin/sh

# HCI -> Exit 0 if successful
python3 ${name}.py  && exit 0

# Exit 1 if unsuccessful
exit 1