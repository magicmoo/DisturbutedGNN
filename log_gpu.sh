#!/bin/bash

# rm -rf ./savedata/gpu.log
rm -rf ./savedata/hostmemory.log
count=0
end=$((60*10))       # 30 seconds
while [ $count -lt $end ]
do
    # nvidia-smi --query-gpu=utilization.gpu --format=csv >> ./savedata/gpu.log
    free -m >> ./savedata/hostmemory.log
    sleep 1
    ((count++))
done