import pytest

# FIXME properly set lib import
import sys
from pathlib import Path

import stlarm.utils
import pandas as pd
import pandas.testing as tm


def test_to_is_holiday_christmas():
    dates = pd.date_range(start="2012-12-24", end="2013-01-02")
    s = pd.Series(dates)

    result = stlarm.utils.to_is_holiday(s)

    expected = pd.Series(
        [
            False,
            True,  # christmas
            False,
            False,
            False,
            False,
            False,
            False,
            True,  # new year
            False,
        ]
    )

    tm.assert_series_equal(result, expected)


def test_to_is_holiday_forth_july():
    dates = pd.to_datetime(["2012-07-04", "2013-07-03", "2014-07-04"])
    s = pd.Series(dates)
    print(s)

    result = stlarm.utils.to_is_holiday(s)

    expected = pd.Series([True, False, True])  # 4th july  # 4th july

    tm.assert_series_equal(result, expected)


def test_to_is_weekday_week():
    dates = pd.date_range(start="2012-01-01", periods=7)  # was a sunday
    s = pd.Series(dates)
    result = stlarm.utils.to_is_weekday(s)

    expected = pd.Series([False, True, True, True, True, True, False])

    tm.assert_series_equal(result, expected)


def test_to_is_weekday_weekends():
    dates = pd.date_range(start="2012-01-01", periods=7, freq="W")  # was a sunday
    s = pd.Series(dates)
    result = stlarm.utils.to_is_weekday(s)

    expected = pd.Series([False] * 7)

    tm.assert_series_equal(result, expected)


def test_to_is_weekday_weekdays():
    dates = pd.bdate_range(start="2012-01-02", periods=7)  # was a sunday
    s = pd.Series(dates)
    result = stlarm.utils.to_is_weekday(s)

    expected = pd.Series([True] * 7)

    tm.assert_series_equal(result, expected)
