#!/usr/bin/env python3
"""Build latex3 Table IV main-task statistics from its manifest."""

from build_latex3_result_tables import main
import sys


if __name__ == "__main__":
    sys.argv += ["--only", "main_task_rows"]
    main()
