from abc import ABC
import io
import logging
import pickle
import re

from owlready2 import *
import pandas as pd
from tqdm import tqdm

from .onto_mapping import (
    CustomOntoMapping,
    CustomRelationsStore,
    OntoMapping,
    STEPOntoMapping,
    step_mapping_to_custom_mapping,
)
from .utils import timed, is_within_time_window, is_within_radius

logger = logging.getLogger("stlarm.onto")


class OntoRepresentation(ABC):
    AVAILABLE_CUSTOM_RELATIONS = {
        "before",
        "withinTimeWindow",
        "withinRadius",
    }

    def __init__(
        self,
        name: str,
        mapping_cls=None,
        iri=None,
        time_granularity=None,
        max_hours_diff=None,
        max_distance_km=None,
        excluded_relations=[],
    ):
        excluded_relations = set(excluded_relations)

        unknown_relations = {
            r
            for r in excluded_relations
            if r not in OntoRepresentation.AVAILABLE_CUSTOM_RELATIONS
        }

        if len(unknown_relations) > 0:
            raise ValueError(
                "excluded_relations contains unknown relations: %s", unknown_relations
            )

        if max_hours_diff is None or max_hours_diff <= 0:
            logger.info(
                f"max_hours_diff is {max_hours_diff}. Disabling withinTimeWindow relation."
            )
            excluded_relations.add("withinTimeWindow")

        if max_distance_km is None or max_distance_km <= 0:
            logger.info(
                f"max_distance_km is {max_distance_km}. Disabling withinRadius relation."
            )
            excluded_relations.add("withinRadius")

        self.name = name

        self.mapping = mapping_cls(iri, time_granularity)

        self.max_hours_diff = max_hours_diff
        self.max_distance_km = max_distance_km

        self.excluded_relations = excluded_relations

        self.custom_relations = CustomRelationsStore(self.excluded_relations)

    @property
    def pretty_id(self):
        return OntoRepresentation._build_pretty_id(
            self.name,
            self.mapping.time_granularity,
            self.max_hours_diff,
            self.max_distance_km,
            self.excluded_relations,
        )

    @staticmethod
    def _build_pretty_id(
        name, time_granularity, max_hours_diff, max_distance_km, excluded_relations
    ):
        return "_".join(
            map(
                str,
                [
                    name,
                    time_granularity,
                    max_hours_diff,
                    max_distance_km,
                    "_".join(excluded_relations) if excluded_relations else "None",
                ],
            )
        )

    def save(self, folder):
        # time granularity, max hours, max distance, ignored relations
        with open(f"{folder}/{self.pretty_id}.pickle", "wb",) as f:
            pickle.dump(self, f)

    @staticmethod
    def load(
        folder,
        name,
        time_granularity=None,
        max_hours_diff=None,
        max_distance_km=None,
        excluded_relations=[],
    ):
        excluded_relations = set(excluded_relations)
        # time granularity, max hours, max distance, ignored relations
        pretty_id = OntoRepresentation._build_pretty_id(
            name, time_granularity, max_hours_diff, max_distance_km, excluded_relations
        )
        with open(f"{folder}/{pretty_id}.pickle", "rb",) as f:
            return pickle.load(f)

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.custom_relations._update_entities_with_ontology(self.onto)

    @classmethod
    def from_representation_and_mapping(
        cls,
        name,
        old_onto_representation: "OntoRepresentation",
        new_mapping: OntoMapping,
    ):
        # TODO should be a better way to create an OntoRepresentation from a mapping
        new_onto_representation = cls(
            name=name,
            time_granularity=old_onto_representation.mapping.time_granularity,
            max_hours_diff=old_onto_representation.max_hours_diff,
            max_distance_km=old_onto_representation.max_distance_km,
            excluded_relations=old_onto_representation.excluded_relations,
        )
        new_onto_representation.mapping = new_mapping
        return new_onto_representation

    def __eq__(self, other):
        if self.max_hours_diff != other.max_hours_diff:
            return False

        if self.max_distance_km != other.max_distance_km:
            return False

        if self.excluded_relations != other.excluded_relations:
            return False

        if self.mapping.time_granularity != other.mapping.time_granularity:
            return False

        return self.get_triples().equals(other.get_triples())

    def get_relations(self):
        return self.get_filtered_triples()["pred"].unique()

    @property
    def onto(self):
        return self.mapping.onto

    @timed(logger.info, "Generating representation done in {}s.")
    def gen_representation(self, trajs, traj_data=None):
        if (
            "lat" not in list(trajs)[0][1].columns
            or "lon" not in list(trajs)[0][1].columns
        ):
            logger.info(
                f"No lat or lon info provided. Disabling withinRadius relation."
            )
            self.excluded_relations.add("withinRadius")

        if self.excluded_relations:
            logger.info("Ignoring following relations: %s", self.excluded_relations)

        with self.onto:
            self._gen_trajs_representation(trajs)

            if traj_data is not None:
                self._gen_traj_data_representation(traj_data)

            self.mapping.add_entity2entity_relations(self.custom_relations)

    def _gen_traj_data_representation(self, traj_data):
        for tid, data in tqdm(traj_data.items(), desc="trajs data"):
            self.mapping.add_trajectory_data(tid, data)

    def _gen_trajs_representation(self, trajs):
        # TODO remove total from tqdm when pandas#33132 gets closed
        # https://github.com/pandas-dev/pandas/issues/33132
        for index, data in tqdm(trajs, desc="trajs", total=trajs.ngroups):
            _, uid = index
            tid = data.tid.values[0]
            self.mapping.add_trajectory(uid, tid)

            self._gen_checkins_representation(tid, data)

    def _gen_checkins_representation(self, tid, checkins):
        # TODO improve function
        checkins = list(checkins.itertuples())
        for checkin in checkins:
            cid = checkin.cid
            self.mapping.add_checkin(tid, cid)
            time = self.mapping._get_time(checkin.local_date_time)

            self.mapping.add_checkin_poi(cid, checkin.venue_name)
            self.mapping.add_checkin_data(cid, {"Time": time})

            self.mapping.add_poi_data(
                checkin.venue_name, {"POICategory": checkin.root_category_name}
            )

        # for each checkin
        for i, curr_checkin_row in enumerate(checkins[:-1]):
            curr_cid = curr_checkin_row.cid
            curr_checkin = self.mapping.get_checkin(curr_cid)

            # for each aftwerards checkin
            for j, next_checkin_row in enumerate(checkins[i + 1 :]):
                next_cid = next_checkin_row.cid
                next_checkin = self.mapping.get_checkin(next_cid)

                self.custom_relations.add_to_relation(
                    "before", curr_checkin, next_checkin
                )

                if is_within_time_window(
                    curr_checkin_row, next_checkin_row, self.max_hours_diff
                ):
                    self.custom_relations.add_to_symmetrical_relation(
                        "withinTimeWindow", curr_checkin, next_checkin
                    )

                if is_within_radius(
                    curr_checkin_row, next_checkin_row, self.max_distance_km
                ):
                    self.custom_relations.add_to_symmetrical_relation(
                        "withinRadius", curr_checkin, next_checkin
                    )

    def get_raw_triples(self):
        s = io.BytesIO()
        logger.debug("Getting onto triples")
        self.onto.save(s, format="ntriples")

        data = s.getvalue().decode("utf-8")
        logger.debug("[DONE] Getting onto triples")
        return data

    def get_triples(self, remove_uri=False):
        data = self.get_raw_triples()

        if remove_uri:
            # remove the "http...#" part of a uri
            # (i.e., letting only the instance name behind),
            # for onto's and step's base iri
            # (i.e., keeping the full uri if it is a w3.org (rdf:type, rdfs:subClassOf, etc)
            ontology_uri_regex = "|".join(
                [self.onto.base_iri]
                + [o.base_iri for o in self.onto.imported_ontologies]
            )
            logger.debug(f"Removing regex {ontology_uri_regex}")
            data = re.sub(ontology_uri_regex, "", data)
            logger.debug(f"[DONE] Removing regex {ontology_uri_regex}")

        wrapper = io.StringIO(data)
        df = pd.read_csv(wrapper, names="sub pred obj .".split(), sep=" ")
        df = df.sort_values(list(df.columns), ascending=False).reset_index(drop=True)
        assert df["."].nunique() == 1, df["."].unique()

        return df

    def get_filtered_triples(self):
        df = self.get_triples(remove_uri=True)

        df = df[~df["pred"].str.startswith("<http://www.w3.org")]
        return df

    def save_filtered_triples(self, file, triples_sampler=None):
        df = self.get_filtered_triples()

        if triples_sampler:
            df = triples_sampler.sample(df)

        self.save_df(file, df)

    def save_triples(self, file):
        df = self.get_triples()
        self.save_df(file, df)

    def save_df(self, file, df):
        # TODO not a good method api/name
        df.to_csv(file, sep="\t", header=False, index=False)


class CustomRepresentation(OntoRepresentation):
    def __init__(self, **kwargs):
        iri = kwargs.pop("iri", "http://custom_data.com")
        super().__init__(mapping_cls=CustomOntoMapping, iri=iri, **kwargs)


class STEPRepresentation(OntoRepresentation):
    def __init__(self, **kwargs):
        iri = kwargs.pop("iri", "http://step_data.com")
        super().__init__(mapping_cls=STEPOntoMapping, iri=iri, **kwargs)

    @timed(logger.info, "STEP to Custom done in {}s.")
    def to_custom_representation(self, name, iri="http://custom_data.com"):
        custom_mapping = step_mapping_to_custom_mapping(iri, self.mapping)
        custom_representation = CustomRepresentation.from_representation_and_mapping(
            name, self, custom_mapping
        )
        return custom_representation
