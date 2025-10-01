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
