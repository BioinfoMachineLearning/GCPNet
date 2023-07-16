#!/bin/bash

for file in gpu_job_*.yaml; do
    kubectl apply -f "$file"
    sleep 2
done