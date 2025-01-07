#!/bin/bash -e

start_id=135969
end_id=135992

for job_id in $(seq $start_id $end_id); do
    scancel $job_id
done
