from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utility.evaluation import parse_args


def main() -> None:
    args = parse_args()

    # Keep CLI startup lightweight so --help does not eagerly import heavy model stacks.
    from src.utility.evaluation import run_evaluation

    run_evaluation(args)


if __name__ == "__main__":
    main()
