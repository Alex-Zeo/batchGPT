import pandas as pd
import pytest

from app.excelreader import ExcelReader


def test_excel_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    def fake_read_excel(*args: object, **kwargs: object) -> pd.DataFrame:
        return df

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    reader = ExcelReader()
    records = reader.read("dummy.xlsx")
    assert records == df.to_dict(orient="records")


def test_excel_reader_invalid_extension() -> None:
    reader = ExcelReader()
    with pytest.raises(ValueError):
        reader.read("file.txt")
