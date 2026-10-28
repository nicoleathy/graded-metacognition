"""Write the frozen protocol record (conditions, decoding, subsets, scales and a hash of the rendered prompts).
Run it once before the sweep; tests fail if protocol.py drifts from the file.

    python experiments/paper1/freeze_protocol.py
"""

import json

from common import ROOT
from protocol import protocol_record

FROZEN = ROOT / "experiments/paper1/frozen/protocol.json"

if __name__ == "__main__":
    record = protocol_record()
    FROZEN.write_text(json.dumps(record, indent=1, default=list) + "\n")
    print("protocol sha256", record["sha256"])
