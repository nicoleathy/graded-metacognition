#!/bin/bash
# Wait for the first sweep queue (run_sweep_queue.sh) to finish, then hand over to the panel queue. A separate file from the queue itself,
# so run_panel_queue.sh can still be edited while this one waits.
cd "$(dirname "$0")/../.."
while pgrep -f "[r]un_sweep_queue.sh" > /dev/null; do sleep 60; done
echo "first queue finished: $(date '+%F %T')"
exec experiments/paper1/run_panel_queue.sh
