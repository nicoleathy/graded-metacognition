"""PROBE_WORDING is read once, when probe_validity.scales is first imported. These tests pin the legacy wording w1; the tests of w2 and w3
start their own processes with the variable set. A wording exported in the shell (a cluster job for w2, say) must not reach this process."""

import os

os.environ.pop("PROBE_WORDING", None)
