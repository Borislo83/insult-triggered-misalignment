#!/usr/bin/env bash
pkill -CONT -f 'em\.lora_run|em\.generate' && echo "resumed" || echo "nothing paused"
