
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_classifier_sea import main

_BACKBONE_DEFAULTS = [
    "--no_structured_evidence",
    "--no_confidence_gate",
    "--no_consistency_loss",
    "--no_relation_prior",
]


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *_BACKBONE_DEFAULTS, *sys.argv[1:]]
    main()