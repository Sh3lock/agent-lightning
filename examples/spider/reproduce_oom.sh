#!/bin/bash
export PYTHONPATH=/root/verl_pass/verl:$PYTHONPATH
cd /root/verl_pass/agent-lightning/examples/spider

# Clean up previous runs
echo "Cleaning up previous processes..."
pkill -f train_sql_agent.py
sleep 2

echo "Starting OOM reproduction with top 48 longest sequences..."
python train_sql_agent.py qwen --config-file configs/reproduce_oom.json
