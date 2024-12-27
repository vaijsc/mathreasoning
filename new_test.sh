#!/bin/bash -e
GPUS_PER_NODE=4
NUM_NODES=2
result=$((64 / (GPUS_PER_NODE * NUM_NODES)))
echo $result

