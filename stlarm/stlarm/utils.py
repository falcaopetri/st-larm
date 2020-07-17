import collections
import datetime
import functools
import time
from typing import Callable, Optional

import geopy.distance

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


def timed(logger_func: Callable[[str], None], message: Optional[str] = None):
    """This decorator logs the execution time for the decorated function."""

    if message is None:
        message = "{func} ran in {}s"

    def log(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.monotonic()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed = time.monotonic() - start
                elapsed = round(elapsed, 2)
                logger_func(message.format(elapsed, func=func.__name__))

        return wrapper

    return log


def millis_to_seconds(x):
    return x / 1000.0


def time_granularity(f):
    def wrapper(self, time):
        granular_time = self.get_granular_time(time)
        return f(self, granular_time)

    return wrapper


def to_is_holiday(s: pd.Series):
    """
    Returns a Series indicating whether the dates are US Federal Holidays
    """
    dates = s.dt.date
    cal = USFederalHolidayCalendar()
    start = dates.min()
    end = dates.max()
    us_holidays = cal.holidays(start=start, end=end)
    us_holidays = [x.date() for x in us_holidays]

    return dates.transform(lambda v: v in us_holidays).rename("is_holiday")


def to_is_weekday(dates: pd.Series):
    """
    Returns a Series indicating whether the dates are weekdays
    """
    is_weekday = dates.dt.dayofweek < 5
    return is_weekday.rename("is_weekday")


def to_day_of_the_week(dates: pd.Series):
    """
    Returns a Series indicating the dates' day of the week
    """
    return dates.dt.day_name().rename("day_of_the_week")


def to_month(dates: pd.Series):
    """
    Returns a Series indicating the dates' month
    """
    return dates.dt.month_name().rename("month")


def is_within_time_window(c1, c2, max_hours_diff=2):
    try:
        diff_hours = abs(c1.local_date_time - c2.local_date_time) / datetime.timedelta(
            hours=1
        )
        return diff_hours <= max_hours_diff
    except AttributeError:
        return None


def is_within_radius(c1, c2, max_distance_km=2):
    if c1.venue_id == c2.venue_id:
        # same venue!
        return False

    try:
        c1_lat_lon = (c1.lat, c1.lon)
        c2_lat_lon = (c2.lat, c2.lon)
    except AttributeError:
        return None

    distance = geopy.distance.distance(c1_lat_lon, c2_lat_lon).km
    return distance <= max_distance_km


def normalize_string_series(series: pd.Series):
    return (
        series.str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")  # remove non-ascii
        .str.decode("utf-8")
        .str.replace(r"[^A-Za-z0-9_ ]", "")  # remove special chars
        .str.strip()
        .str.title()  # capitalize each word
        .str.replace(" ", "")  # remove spaces
    )


def save_rules(rules, filename="mined_rules.txt"):
    content = [r.getFullRuleString() for r in rules]
    content = "\n".join(content)

    with open(filename, "w") as f:
        f.write(content)

    content = [r.getRuleString() for r in rules]
    content = "\n".join(content)

    with open(filename + ".nometric", "w") as f:
        f.write(content)


# Source: https://stackoverflow.com/a/17718729
def get_system_memory_info():
    """
    Get node total memory and memory usage
    """

    def kb_to_gb(x):
        return x / (1024.0 ** 2)

    with open("/proc/meminfo", "r") as mem:
        ret = {}
        tmp = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) == "MemTotal:":
                ret["total"] = int(sline[1])
            elif str(sline[0]) in ("MemFree:", "Buffers:", "Cached:"):
                tmp += int(sline[1])
        ret["total"] = kb_to_gb(ret["total"])
        ret["free"] = kb_to_gb(tmp)
        ret["used"] = ret["total"] - ret["free"]
    return ret
