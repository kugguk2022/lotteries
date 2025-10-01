from __future__ import annotations

from typing import List

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series


class EuroMillionsSchema(pa.SchemaModel):
    draw_date: Series[pd.Timestamp] = pa.Field(nullable=False)
    ball_1: Series[int] = pa.Field(ge=1, le=50)
    ball_2: Series[int] = pa.Field(ge=1, le=50)
    ball_3: Series[int] = pa.Field(ge=1, le=50)
    ball_4: Series[int] = pa.Field(ge=1, le=50)
    ball_5: Series[int] = pa.Field(ge=1, le=50)
    star_1: Series[int] = pa.Field(ge=1, le=12)
    star_2: Series[int] = pa.Field(ge=1, le=12)

    class Config:
        coerce = True  # cast types on validate


EXPECTED_COLUMNS: List[str] = [
    "draw_date",
    "ball_1",
    "ball_2",
    "ball_3",
    "ball_4",
    "ball_5",
    "star_1",
    "star_2",
]


def validate_df(df: pd.DataFrame) -> DataFrame[EuroMillionsSchema]:
    """Validate draw data structure and ensure timezone-naive timestamps."""

    missing = [name for name in EXPECTED_COLUMNS if name not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    coerced = df.copy()
    coerced["draw_date"] = pd.to_datetime(coerced["draw_date"], utc=True).dt.tz_convert(None)
    return EuroMillionsSchema.validate(coerced)
