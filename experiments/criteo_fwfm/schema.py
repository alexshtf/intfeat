from __future__ import annotations

LABEL_COLUMN = "label"
INTEGER_COLUMNS = [f"I{i}" for i in range(1, 14)]
CATEGORICAL_COLUMNS = [f"C{i}" for i in range(1, 27)]
ALL_COLUMNS = [LABEL_COLUMN, *INTEGER_COLUMNS, *CATEGORICAL_COLUMNS]
