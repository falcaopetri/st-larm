from abc import ABC, abstractmethod
from datetime import timedelta
import logging
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset

from trajminer.datasets import load_foursquare_checkins
from stlarm.utils import (
    timed,
    to_is_holiday,
    to_is_weekday,
    to_day_of_the_week,
    to_month,
    normalize_string_series,
)

STLARM_BASE_DIR = Path(os.getenv("STLARM_BASE_DIR", ""))
logger = logging.getLogger("stlarm.data")


class TrajData(ABC):
    def __init__(self, apply_filtering=True):
        self._df = type(self)._load_df()
        self._data = None
        self._trajectories = None
        self.apply_filtering = apply_filtering

    @staticmethod
    @abstractmethod
    def _load_df():
        pass

    def _process_data(self):
        pass

    @property
    def df(self):
        return self._df

    @property
    def data(self):
        if self._data is None:
            self._process_data()
        return self._data

    @property
    def trajectories(self):
        if self._trajectories is None:
            self._process_trajectories()
        return self._trajectories

    @property
    def cids(self):
        return self.data["cid"].unique()

    @property
    def tids(self):
        return self.data["tid"].unique()

    @property
    def uids(self):
        return self.data["uid"].unique()

    def get_trajectories(self, tids=None, cids=None):
        if tids is None:
            return self.trajectories
        if cids is None:
            return self.cids

        tids = set(tids)
        cids = set(cids)
        filtered_data = self.data[
            self.data["tid"].isin(tids) & self.data["cid"].isin(cids)
        ]
        return self._apply_groupby(filtered_data)

    def _get_data_for_tids(self, tids):
        tids = set(tids)
        df = self.data.copy()
        return df[df["tid"].isin(tids)]

    @abstractmethod
    def get_trajectories_categories(self, tids=None) -> dict:
        pass

    def get_all_data(self, tids=None, uids=None, cids=None):
        """Returns data, trajectories, and traj-categories"""
        arg_tids = tids
        arg_uids = uids
        arg_cids = cids

        if tids is None:
            tids = self.tids

        if uids is None:
            uids = self.uids

        if cids is None:
            cids = self.cids

        tids = set(tids)
        uids = set(uids)
        cids = set(cids)

        # in case we have specified a subset of of tids and/or uids,
        # get the intersection of them
        data = self.data
        data = data[
            data["tid"].isin(tids) & data["uid"].isin(uids) & data["cid"].isin(cids)
        ]
        tids = data["tid"].unique()
        uids = data["uid"].unique()
        cids = data["cid"].unique()

        if len(tids) == 0 or len(uids) == 0 or len(cids) == 0:
            raise ValueError(
                f"No data for specified tids={arg_tids}, uids={arg_uids}, cids={arg_cids}."
            )

        trajectories = self.get_trajectories(tids, cids)
        trajectories_categories = self.get_trajectories_categories(tids)

        return data, trajectories, trajectories_categories

    def _apply_groupby(self, df):
        return df.groupby(by=[pd.Grouper(key="local_date_time", freq="D"), "uid"])

    def _process_trajectories(self):
        self._trajectories = self._apply_groupby(self.data)


class FoursquareData(TrajData, ABC):
    def get_trajectories_categories(self, tids=None):
        if tids is None:
            tids = self.tids

        tids_data = self._get_data_for_tids(tids)

        # get series with datetime indexed by tid
        traj_dates = pd.Series(
            data=tids_data["local_date_time"].values,
            index=tids_data["tid"].values,
            copy=True,
        )
        # discard duplicated tid
        traj_dates = traj_dates[~traj_dates.index.duplicated(keep="first")]

        holiday = to_is_holiday(traj_dates).map({True: "Holiday", False: np.nan})
        weekday = to_is_weekday(traj_dates).map({True: "Weekday", False: "Weekend"})

        traj_data = pd.concat(
            [to_day_of_the_week(traj_dates), to_month(traj_dates)], axis=1
        )

        traj_data = traj_data.rename(
            columns={"day_of_the_week": "DayOfTheWeek", "month": "Month"}
        )

        # TODO improve performance
        traj_data["TrajectoryCategory"] = pd.concat([holiday, weekday], axis=1).apply(
            lambda x: x.dropna().tolist(), axis=1
        )

        return traj_data.to_dict(orient="index")

    @timed(logger.debug, "Filtering data done in {}s.")
    def filter_data(self, df):
        # TODO create parameterizable API for data filtering
        # TODO improve performance
        logger.debug("Starting data filtering...")

        def filter_x_per_y(df, at_least, x, per):
            """filters data that have at least $at_least $x unique values per $per"""
            return df.groupby(per, as_index=False, sort=False).filter(
                lambda g: g[x].nunique() >= at_least
            )

        # Optimization: let's drop the rare venues so we speed up the duplicated check-ins check below
        # Of course, we *must* apply the same filter *after* the duplicated filter
        # filter venues with fewer than 5 checkins
        df = filter_x_per_y(df, at_least=5, x="cid", per="venue_id")

        # filter duplicated check-ins
        # i.e. same user at the same place within a 10-min threshold
        df = df.groupby(["uid", "venue_id"], as_index=False, sort=False).apply(
            lambda g: g[
                g["local_date_time"].diff().fillna(pd.Timedelta(minutes=20))
                > timedelta(minutes=10)
            ]
        )

        # filter venues with fewer than 5 checkins
        df = filter_x_per_y(df, at_least=5, x="cid", per="venue_id")
        # filter trajs with fewer than 5 checkins
        df = filter_x_per_y(df, at_least=5, x="cid", per="tid")
        # filter users with fewer than 10 trajs
        df = filter_x_per_y(df, at_least=10, x="tid", per="uid")

        return df

    @timed(logger.info, "Processing data done in {}s.")
    def _process_data(self):
        # set _u_ser id
        df = (
            self.df.data.reset_index()
            .drop(columns="pid")
            .rename(columns={"tid": "uid"})
        )

        # set utc_date_time as correct type
        df = df.assign(utc_date_time=pd.to_datetime(df["utc_date_time"]))

        # apply offset to utc_date_time so we have local_date_time info
        local_date_time = (
            df.groupby("utc_offset_min", as_index=False)
            .apply(lambda g: DateOffset(minutes=g.name).apply_index(g["utc_date_time"]))
            .droplevel(0)
            .rename("local_date_time")
        )

        # set local_date_time
        df = (
            df.merge(local_date_time, left_index=True, right_index=True)
            .drop(columns=["utc_offset_min", "utc_date_time"])
            .sort_values(["uid", "local_date_time"])
        )

        with open("data_resources/venue_id_to_name_id.json", "r") as f:
            venue_id_to_name = json.load(f)
        df["venue_name"] = df["venue_id"].map(lambda x: venue_id_to_name.get(x, x))

        with open("data_resources/category_id_to_root_category_name.json", "r") as f:
            category_id_to_root_category_name = json.load(f)

        # There is an old "Ferry" category which we need to update to the "Boat or Ferry" category
        ferry_id = "4e51a0c0bd41d3446defbb2e"
        boat_or_ferry_id = "4bf58dd8d48988d12d951735"
        df.loc[df["category_id"] == ferry_id, "category_id"] = boat_or_ferry_id
        # Now we succesfully convert all categories
        df["root_category_name"] = df["category_id"].map(
            category_id_to_root_category_name
        )

        # normalize category_name
        df["category_name"] = normalize_string_series(df["category_name"])
        df["root_category_name"] = normalize_string_series(df["root_category_name"])

        # create mapping from date to sequential id
        dates = df["local_date_time"].dt.date
        unique_dates = dates.sort_values().unique()
        dates_to_id = {date: id_ for id_, date in enumerate(unique_dates)}

        # create tid as uid + date_id
        tid = df["uid"].astype(str) + "_" + dates.map(dates_to_id).astype(str)
        df = df.assign(tid=tid)

        # create cid as uid + date_id + sequential id
        cid = df["tid"] + "_" + df.groupby("tid").cumcount().astype(str)
        df = df.assign(cid=cid).sort_values(["uid", "tid", "cid"])

        if self.apply_filtering:
            # filter data
            df = self.filter_data(df)

        df = df.reset_index(drop=True)

        if df.isnull().any().any():
            raise ValueError(
                f"Found NaN in the columns marked as True:\n {df.isnull().any()}"
            )

        self._data = df


class NYCFoursquareData(FoursquareData):
    @staticmethod
    def _load_df():
        return load_foursquare_checkins("nyc")
