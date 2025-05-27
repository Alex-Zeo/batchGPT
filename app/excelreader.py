from __future__ import annotations

from pathlib import Path
from typing import IO, Any, Dict, List, Union, cast

import pandas as pd

from .logger import logger


class ExcelReader:
    """Read Excel spreadsheets and return structured data."""

    def read(self, source: Union[str, Path, IO[bytes]]) -> List[Dict[str, Any]]:
        """Load an Excel file using pandas.

        Parameters
        ----------
        source:
            Path to the Excel file or a binary file-like object. Only ``.xlsx``
            and ``.xls`` files are supported.

        Returns
        -------
        List[Dict[str, Any]]
            The rows of the spreadsheet as dictionaries.

        Raises
        ------
        ValueError
            If the file extension is not supported.
        IOError
            If pandas fails to read the file.
        """
        if isinstance(source, (str, Path)):
            ext = Path(source).suffix.lower()
        else:
            ext = Path(getattr(source, "name", "")).suffix.lower()

        if ext not in {".xlsx", ".xls"}:
            raise ValueError(f"Unsupported file type: {ext}")

        try:
            df = pd.read_excel(source)
        except FileNotFoundError as exc:
            logger.error(f"Excel file not found: {exc}")
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to read Excel file: {exc}")
            raise IOError(f"Failed to read Excel file: {exc}") from exc

        data = cast(List[Dict[str, Any]], df.to_dict(orient="records"))
        logger.info(
            f"Loaded Excel data with {len(data)} rows and {len(df.columns)} columns"
        )
        return data
