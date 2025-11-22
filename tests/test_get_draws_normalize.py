from euromillions.get_draws import normalize


def test_normalize_parses_minimal_csv():
    csv_text = (
        "Date,Ball1,Ball2,Ball3,Ball4,Ball5,Lucky Star1,Lucky Star2\n"
        "2024-01-02,1,2,3,4,5,1,2\n"
        "2024-01-09,6,7,8,9,10,3,4\n"
    )

    df = normalize(csv_text)

    assert len(df) == 2
    assert df.iloc[0]["ball_1"] == 1
    assert df.iloc[1]["star_2"] == 4


def test_normalize_dedupes_and_handles_whitespace():
    csv_text = (
        " Date ,Ball 1 ,Ball 2 ,Ball 3 ,Ball 4 ,Ball 5 ,Lucky Star 1 ,Lucky Star 2 ,Ignore\n"
        "2024-01-09 ,6,7,8,9,10,3,4,foo\n"
        "2024-01-02 ,1,2,3,4,5,1,2,bar\n"
        "2024-01-09 ,6,7,8,9,10,3,4,baz\n"
    )

    df = normalize(csv_text)

    assert list(df.draw_date.astype(str)) == ["2024-01-02", "2024-01-09"]
    assert df.iloc[-1]["ball_5"] == 10
