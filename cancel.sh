#!/bin/bash -e

start_id=140392
end_id=140398

for job_id in $(seq $start_id $end_id); do
    scancel $job_id
done
