#!/bin/bash

for file in train_*.bash; do
    bsub "$file"
done