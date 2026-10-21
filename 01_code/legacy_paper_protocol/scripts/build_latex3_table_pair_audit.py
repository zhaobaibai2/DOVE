#!/usr/bin/env python3
"""Build latex3 Table VIII actor-prior audit statistics from its manifest."""

from build_latex3_result_tables import main
import sys


if __name__ == "__main__":
    sys.argv += ["--only", "pair_audit_rows"]
    main()
