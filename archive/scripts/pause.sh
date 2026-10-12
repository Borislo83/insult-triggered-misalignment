#!/usr/bin/env bash
# Freeze the running GPU job (training/generation) so the laptop is usable; resume with resume.sh.
# Memory the job holds (~7 GB) stays reserved, so keep other use light.
pkill -STOP -f 'em\.lora_run|em\.generate' && echo "paused" || echo "nothing running to pause"
