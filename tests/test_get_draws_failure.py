import pandas as pd
import pytest
import requests

from euromillions.get_draws import (
    FetchError,
    NormalizationError,
    fetch_raw_csv,
    normalize,
    _cache_key,  # type: ignore[attr-defined]
)


class _FailingSession:
    def get(self, *args, **kwargs):
        raise requests.ConnectionError("boom")


def test_fetch_uses_cache_on_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("EUROMILLIONS_CACHE_DIR", str(tmp_path))
    from euromillions.get_draws import PRIMARY_URL  # local import to avoid cycle

    cache_file = _cache_key(PRIMARY_URL, {}, tmp_path)
    cache_file.write_text("Date,Ball1,Ball2,Ball3,Ball4,Ball5,Lucky Star1,Lucky Star2\n", encoding="utf-8")

    text = fetch_raw_csv(session=_FailingSession())

    assert "Ball1" in text


def test_fetch_raises_when_no_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("EUROMILLIONS_CACHE_DIR", str(tmp_path))
    with pytest.raises(FetchError):
        fetch_raw_csv(session=_FailingSession(), use_cache=False)


def test_normalize_rejects_bad_shape():
    bad_csv = "Date,Ball1,Ball2\n2024-01-01,1,2"
    with pytest.raises(NormalizationError):
        normalize(bad_csv)


def test_normalize_dedupes_duplicates():
    csv_text = (
        "Date,Ball1,Ball2,Ball3,Ball4,Ball5,Lucky Star1,Lucky Star2\n"
        "2024-01-02,1,2,3,4,5,1,2\n"
        "2024-01-02,1,2,3,4,5,1,2\n"
    )

    df = normalize(csv_text)

    assert len(df) == 1
    assert pd.to_datetime(df.iloc[0]["draw_date"]).date().isoformat() == "2024-01-02"
