#!/bin/bash
# Drives the whole reproduction: waits for story generation, then runs
# extraction -> geometry/validation -> steering, logging to pipeline.log.
cd /home/carlo/emotion-vectors-repro || exit 1
PY=/home/carlo/neuron-surgeon/.venv/bin/python
LOG=pipeline.log

echo "=== driver start $(date) ===" >> $LOG
while pgrep -f "[e]v1_gen.py" > /dev/null; do sleep 20; done
echo "=== generation finished $(date) ===" >> $LOG

$PY ev2_extract.py >> $LOG 2>&1
echo "=== extract rc=$? $(date) ===" >> $LOG

$PY ev3_analyze.py >> $LOG 2>&1
echo "=== analyze rc=$? $(date) ===" >> $LOG

$PY ev4_steer.py --rollouts 20 --steer-emotions 20 >> $LOG 2>&1
echo "=== steer rc=$? $(date) ===" >> $LOG

echo "=== PIPELINE COMPLETE $(date) ===" >> $LOG