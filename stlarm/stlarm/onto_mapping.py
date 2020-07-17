from abc import ABC, abstractmethod
from collections import defaultdict
import datetime
from functools import lru_cache
import logging
from typing import Union
import uuid

from owlready2 import *
from tqdm import tqdm

from .utils import time_granularity, timed


logger = logging.getLogger("stlarm.onto_mapping")


class CustomRelationsStore:
    # FIXME we depend on the class owner to call _update_entities_with_ontology during unpickle

    def __init__(self, excluded_relations=[]):
        self.excluded_relations = excluded_relations
        self.custom_relations = set()

    def __len__(self):
        return len(self.custom_relations)

    def __iter__(self):
        return iter(self.custom_relations)

    def update(self, other: Union["CustomRelationsStore", set]):
        self.custom_relations.update(other)

    def add_to_relation(self, relation, domain, range):
        if relation not in self.excluded_relations:
            self.custom_relations.add((relation, domain, range))

    def add_to_symmetrical_relation(self, relation, domain, range):
        self.add_to_relation(relation, domain, range)
        self.add_to_relation(relation, range, domain)

    def _update_entities_with_ontology(self, onto):
        """Updates entities with they belong to the given onto"""
        custom_relations = set()

        for (
            relation,
            (domain_namespace, domain_name),
            (range_namespace, range_name),
        ) in self.custom_relations:
            if domain_namespace == onto.base_iri:
                domain = onto[domain_name]
            else:
                domain = (domain_namespace, domain_name)

            if range_namespace == onto.base_iri:
                range = onto[range_name]
            else:
                range = (range_namespace, range_name)
            custom_relations.add((relation, domain, range))

        self.custom_relations = custom_relations

    def __getstate__(self):
        state = self.__dict__.copy()
        state["custom_relations"] = {
            (
                relation,
                (domain.namespace.base_iri, domain.name),
                (range.namespace.base_iri, range.name),
            )
            for relation, domain, range in state["custom_relations"]
        }

        return state


class OntoMapping(ABC):
    TIME_GRANULARITIES = [None, "hour", "period"]

    @staticmethod
    def _create_new_class(class_name, base_classes=None):
        # http://owlready.8326.n8.nabble.com/How-to-create-Object-and-Data-Properties-dynamically-td244.html
        if base_classes is None:
            base_classes = (Thing,)
        return types.new_class(class_name, base_classes)

    @staticmethod
    def _create_new_property(prop, domain, range):
        # http://owlready.8326.n8.nabble.com/How-to-create-Object-and-Data-Properties-dynamically-td244.html
        prop = types.new_class(prop, (ObjectProperty,))
        prop.domain = domain
        prop.range = range

    @lru_cache(maxsize=32)
    def _get_or_create_class(self, class_name, base_classes=None):
        try:
            class_ = getattr(self.onto, class_name)
            assert class_ is not None
        except AssertionError:
            # failed to get class once, try to create it
            OntoMapping._create_new_class(class_name, base_classes)
            class_ = getattr(self.onto, class_name)

        if class_ is None:
            # failed to get class twice
            raise RuntimeError(
                f"Failed to create class {class_name}. Base classes: {base_classes}"
            )

        return class_

    def _get_or_create_relation(self, instance, relation, domain, range):
        if not hasattr(instance, relation):
            OntoMapping._create_new_property(relation, domain, range)

        try:
            prop = getattr(instance, relation)
        except AttributeError as e:
            raise RuntimeError(
                f"Failed to create relation {relation}. Domain: {domain}. Range: {range}. Target instance: {instance}"
            ) from e

        return prop

    def __init__(self, iri, time_granularity):
        self.world = World()
        self.cached_uuid = uuid.uuid4()

        self.onto = self.world.get_ontology(iri)
        self.step = self.world.get_ontology(STEPOntoMapping.STEP_URL).load()
        self.onto.imported_ontologies.append(self.step)

        assert time_granularity in OntoMapping.TIME_GRANULARITIES
        self.time_granularity = time_granularity
        self._custom_relations = CustomRelationsStore()

    def __getstate__(self):
        # Source: https://docs.python.org/3/library/pickle.html#pickle-state
        self.world.set_backend(
            filename=f"cache/{self.cached_uuid}.sqlite3"  # , exclusive=False
        )

        # save ontologies to disk
        self.world.save()

        state = self.__dict__.copy()

        # Remove the unpicklable entries
        del state["world"]

        # reassign ontologies
        state["onto"] = state["onto"].base_iri
        state["step"] = state["step"].base_iri

        return state

    def __setstate__(self, state):
        # Source: https://docs.python.org/3/library/pickle.html#pickle-state
        cached_uuid = state.pop("cached_uuid")
        # Restore instance attributes
        self.__dict__.update(state)

        # create world and connect to physical database data
        self.world = World()
        self.world.set_backend(
            filename=f"cache/{cached_uuid}.sqlite3"
        )  # , exclusive=False)

        # loaded self.onto and self.step are the base_iri's,
        # let's get the real ontology reference
        self.onto = self.world.get_ontology(self.onto)
        self.step = self.world.get_ontology(self.step)
        self._custom_relations._update_entities_with_ontology(self.onto)
        self._custom_relations._update_entities_with_ontology(self.step)

    def get_granular_time(self, time):
        if self.time_granularity is None:
            if isinstance(time, datetime.datetime):
                return time.time()
            else:
                return time
        elif self.time_granularity == "hour":
            return time.hour
        elif self.time_granularity == "period":
            # "Specifically, we map check-in time in a day onto three commonly-used
            # time slots in LBSNs, namely morning (8:00-12:00),
            # afternoon(12:00-20:00) and evening/night (20:00-8:00) [47]."
            # Source: "PrivCheck: privacy-preserving check-in data
            # publishing for personalized location based services"
            # https://dl.acm.org/doi/10.1145/2971648.2971685
            periods = "Night", "Morning", "Afternoon", "Night"
            bins = [8, 12, 20, 24]
            for p, b in zip(periods, bins):
                if time.hour < b:
                    return p
        else:
            raise ValueError()

    @abstractmethod
    def add_user(self, uid: str):
        pass

    @abstractmethod
    def add_poi(self, poi_id: str):
        pass

    @abstractmethod
    def _get_trajectory(self, tid: str):
        pass

    @abstractmethod
    def add_checkin(self, tid: str, cid: str):
        pass

    @abstractmethod
    def get_checkin(self, cid: str):
        pass

    @abstractmethod
    def add_checkin_poi(self, cid: str, poi_id: str):
        pass

    @abstractmethod
    def add_checkin_data(self, cid: str, data: dict):
        pass

    def add_user_data(self, uid: str, data: dict):
        user = self.add_user(uid)
        self._add_instance_data(user, data)

    def add_poi_data(self, poi_id: str, data: dict):
        poi = self.add_poi(poi_id)
        self._add_instance_data(poi, data)

    def add_trajectory(self, uid: str, tid: str):
        user = self.add_user(uid)
        traj = self._get_trajectory(tid)
        user.hasTrajectory.append(traj)
        return traj

    def add_trajectory_data(self, tid: str, data: dict):
        traj = self._get_trajectory(tid)
        self._add_instance_data(traj, data)

    @abstractmethod
    def _add_instance_data(self, instance, data: dict):
        pass

    @abstractmethod
    def add_entity2entity_relations(self, relations_store: CustomRelationsStore):
        pass

    def _add_custom_relations(self, relations_store: Union[CustomRelationsStore, set]):
        for relation, domain, range in relations_store:
            prop = self._get_or_create_relation(
                domain, relation, type(domain), type(range)
            )

            prop.append(range)
        self._custom_relations.update(relations_store)

    @time_granularity
    def _get_time(self, time):
        return time


def build_custom_onto(onto):
    with onto:

        class User(Thing):
            pass

        class Trajectory(Thing):
            pass

        class Checkin(Thing):
            pass

        class POI(Thing):
            pass

        class hasTrajectory(User >> Trajectory):
            pass

        class hasCheckin(Trajectory >> Checkin):
            pass

        class hasPOI(Checkin >> POI):
            pass


class CustomOntoMapping(OntoMapping):
    def __init__(self, iri, time_granularity):
        super().__init__(iri, time_granularity)
        build_custom_onto(self.onto)

    def add_user(self, uid: str):
        return self.onto.User(f"User_{uid}")

    def add_poi(self, poi_id: str):
        return self.onto.POI(f"Poi_{poi_id}")

    def _get_trajectory(self, tid: str):
        return self.onto.Trajectory(f"Traj_{tid}")

    def add_checkin(self, tid: str, cid: str):
        uid, tid, _ = cid.split("_")
        tid = f"{uid}_{tid}"

        traj = self._get_trajectory(tid)
        checkin = self.get_checkin(cid)
        traj.hasCheckin.append(checkin)
        return checkin

    def get_checkin(self, cid: str):
        return self.onto.Checkin(f"Checkin_{cid}")

    def add_checkin_poi(self, cid: str, poi_id: str):
        checkin = self.get_checkin(cid)
        poi = self.add_poi(poi_id)
        checkin.hasPOI.append(poi)

    def add_checkin_data(self, cid: str, data: dict):
        checkin = self.get_checkin(cid)
        self._add_instance_data(checkin, data)

    def add_entity2entity_relations(self, relations_store: CustomRelationsStore):
        # TODO maybe we could add a tqdm here?
        self._add_custom_relations(relations_store)

    def _add_instance_data(self, instance, data: dict):
        for class_name, value in data.items():
            class_ = self._get_or_create_class(class_name)
            relation = f"has{class_name}"
            prop = self._get_or_create_relation(
                instance, relation, type(instance), class_
            )

            if not isinstance(value, list):
                value = [value]

            for v in value:
                inst = class_(v)
                if inst not in prop:
                    prop.append(inst)


class STEPOntoMapping(OntoMapping):
    STEP_URL = "https://raw.githubusercontent.com/talespaiva/step/gh-pages/ontology/step_v2.rdf"

    def __init__(self, iri, time_granularity):
        super().__init__(iri, time_granularity)
        # self.step = self.world.get_ontology(STEPOntoMapping.STEP_URL).load()
        # self.onto.imported_ontologies.append(self.step)

    def add_user(self, uid: str):
        return self.step.Agent(f"User_{uid}")

    def add_poi(self, poi_id: str):
        poi_cls = self._get_or_create_class(
            "POI", base_classes=(self.step.ContextualElement,)
        )
        return poi_cls(f"Poi_{poi_id}")

    def _get_trajectory(self, tid: str):
        return self.step.Trajectory(f"Traj_{tid}")

    def add_checkin(self, tid: str, cid: str):
        uid, tid, _ = cid.split("_")
        tid = f"{uid}_{tid}"

        traj = self._get_trajectory(tid)

        # get FOI_Checkin class
        foi_cls_name = "FOI_Checkin"
        foi_cls = self._get_or_create_class(
            foi_cls_name, base_classes=(self.step.FeatureOfInterest,)
        )

        # get FOI_Checkin instance
        foi_name = f"{foi_cls_name}_{tid}"
        foi = foi_cls(foi_name)

        if foi not in traj.hasFeature:
            traj.hasFeature.append(foi)

        checkin = self.get_checkin(cid)
        foi.hasEpisode.append(checkin)

        return checkin

    def get_checkin(self, cid: str):
        return self.step.Episode(f"Checkin_{cid}")

    def add_checkin_poi(self, cid: str, poi_id: str):
        checkin = self.get_checkin(cid)
        poi = self.add_poi(poi_id)
        checkin.relatesTo.append(poi)

    def add_checkin_data(self, cid: str, data: dict):
        checkin = self.get_checkin(cid)

        for contextual_element_name, contextual_data in data.items():
            # get contextual element class
            contextual_element_cls = self._get_or_create_class(
                contextual_element_name, base_classes=(self.step.ContextualElement,)
            )
            # get contextual element instance
            contextual_element = contextual_element_cls(
                f"{contextual_element_name}_{checkin.name}"
            )
            checkin.relatesTo.append(contextual_element)
            self._add_instance_data(
                contextual_element, {contextual_element_name: contextual_data}
            )

    def _add_instance_data(self, instance, data: dict):
        for foi_cls_name, value in data.items():
            # create a FOI with the given name
            foi_cls_name = f"FOI_{foi_cls_name}"
            foi_cls = self._get_or_create_class(
                foi_cls_name, base_classes=(self.step.FeatureOfInterest,)
            )
            # get FOI instance
            foi_name = f"{foi_cls_name}_{instance.name}"
            foi = foi_cls(foi_name)

            if foi not in instance.hasFeature:
                instance.hasFeature.append(foi)

            if not isinstance(value, list):
                value = [value]

            for v in value:
                desc = self.step.QualitativeDescription(v)
                foi.hasSemanticDescription.append(desc)

    def add_entity2entity_relations(self, relations_store: CustomRelationsStore):
        # get CustomRelations class which is a subclass of ContextualElement
        contextual_element_cls = self._get_or_create_class(
            "CustomRelations", base_classes=(self.step.ContextualElement,)
        )

        add_as_custom_relation = defaultdict(set)
        for relation, domain, range in tqdm(relations_store, desc="entity2entity"):
            # We only support domains and range of type Episode.
            # Otherwise we need to add as custom_relation
            if (
                type(domain) is not self.step.Episode
                or type(range) is not self.step.Episode
            ):
                add_as_custom_relation[relation].add((relation, domain, range))
            else:
                # create a CustomRelations instance of the given domain instance
                contextual_element_name = f"CR_{domain.name}"
                contextual_element = contextual_element_cls(contextual_element_name)

                # get FOI cls
                foi_cls_name = f"FOI_{relation}"
                foi_cls = self._get_or_create_class(
                    foi_cls_name, base_classes=(self.step.FeatureOfInterest,)
                )

                # get FOI instance for the given contextual element
                foi_name = f"{foi_cls_name}_{contextual_element_name}"
                foi = foi_cls(foi_name)

                if contextual_element not in domain.relatesTo:
                    domain.relatesTo.append(contextual_element)

                if foi not in contextual_element.hasFeature:
                    contextual_element.hasFeature.append(foi)

                if range not in foi.hasEpisode:
                    foi.hasEpisode.append(range)

        for relation, relation_data in add_as_custom_relation.items():
            logger.info(f"Adding custom relation {relation}.")
            self._add_custom_relations(relation_data)


def step_mapping_to_custom_mapping(iri, step_mapping):
    def process_contextual_element_foi(custom_parent_episode, contextual_element, foi):
        # TODO should not require treating custom_foi differently
        custom_foi = map_step_to_custom_entity(foi)
        for episode in foi.hasEpisode:
            entity2entity_relations.add_to_relation(
                custom_foi[3:],  # remove "has" from relation name
                custom_parent_episode,
                map_step_to_custom_entity(episode),
            )

        ancestors = type(contextual_element).ancestors()
        for semantic_desc in foi.hasSemanticDescription:
            # TODO we could have a way to differentiate first- vs second-class
            # maybe a special taxonomy in the T-box?
            if step_mapping.onto["POI"] in ancestors:
                # should be "first-class" citizen, i.e., a concrete ContextualElement
                custom_contextual_element = map_step_to_custom_entity(
                    contextual_element
                )
                entity2entity_relations.add_to_relation(
                    "has" + type(custom_contextual_element).name,
                    custom_parent_episode,
                    custom_contextual_element,
                )
                process_semantic_description(
                    custom_contextual_element, custom_foi, semantic_desc
                )
            else:
                # should be "second-class" citizen, i.e., a relation
                process_semantic_description(
                    custom_parent_episode, custom_foi, semantic_desc
                )

    def process_contextual_element(episode, contextual_element):
        for foi in contextual_element.hasFeature:
            process_contextual_element_foi(episode, contextual_element, foi)

    def process_episode(custom_parent_instance, custom_foi, episode):
        custom_episode = map_step_to_custom_entity(episode)
        entity2entity_relations.add_to_relation(
            custom_foi, custom_parent_instance, custom_episode
        )

        if len(episode.hasSemanticDescription) > 0:
            raise ValueError("We do not support Episode's semantic description by now.")

        for contextual_element in episode.relatesTo:
            process_contextual_element(custom_episode, contextual_element)

    def process_semantic_description(custom_instance, custom_foi, semantic_desc):
        # TODO should not require treating custom_foi differently
        custom_semantic_desc = map_step_to_custom_entity(
            semantic_desc, custom_foi[3:]  # remove "has" from class name
        )
        entity2entity_relations.add_to_relation(
            custom_foi, custom_instance, custom_semantic_desc
        )

    def process_foi(custom_trajectory, foi):
        custom_foi = map_step_to_custom_entity(foi)

        for episode in foi.hasEpisode:
            process_episode(custom_trajectory, custom_foi, episode)

        for semantic_desc in foi.hasSemanticDescription:
            process_semantic_description(custom_trajectory, custom_foi, semantic_desc)

    def process_trajectory(custom_agent, traj):
        # map trajectory
        custom_trajectory = map_step_to_custom_entity(traj)
        entity2entity_relations.add_to_relation(
            "hasTrajectory", custom_agent, custom_trajectory
        )

        for foi in traj.hasFeature:
            process_foi(custom_trajectory, foi)

    def process_agent(agent):
        # map agent
        custom_agent = map_step_to_custom_entity(agent)

        for trajectory in agent.hasTrajectory:
            process_trajectory(custom_agent, trajectory)

    @lru_cache(maxsize=32)
    def get_known_superclass(mapping, class_):
        try:
            # try to find if some instance's class is subclass of a known class
            # (for which we have a mapping)
            mapped_class = next(x for x in mapping if issubclass(class_, x))
            return mapped_class
        except StopIteration:
            raise ValueError("could not map class %s", class_)

    def map_step_to_custom_entity(instance, new_class_name=None):
        instance_class = type(instance)
        # try to get the mapping of instance's class
        try:
            mapper = direct_map[instance_class]
            return mapper(instance.name)
        except KeyError:
            pass

        # We are dealing with a not already known mapping,
        # so we will need to build the mapper based on the instance's superclass.

        # Let's get the class for which we know how to map
        known_classes = tuple(mapper_builder_fns.keys())
        known_superclass = get_known_superclass(known_classes, instance_class)

        # Let's get the mapper function
        mapper_fn = mapper_builder_fns[known_superclass]

        if new_class_name is None:
            # if the new class name was not specified, we use the name
            # of the instance's class
            new_class_name = instance_class.name

        # we build the mapper passing the new class name
        mapper = mapper_fn(new_class_name)

        # Let's store this mapping (instance_class -> mapper) for future attempts
        direct_map[instance_class] = mapper
        return mapper(instance.name)

    custom_mapping = CustomOntoMapping(iri, step_mapping.time_granularity)
    entity2entity_relations = CustomRelationsStore()
    custom_relations = CustomRelationsStore()

    # List of classes for which we know how to map from instance step:class to custom:class.
    # Function map_step_to_custom_entity will potentially add new known mappings as they are
    # built at runtime.
    direct_map = {
        step_mapping.step.Agent: custom_mapping.onto.User,
        step_mapping.step.Trajectory: custom_mapping.onto.Trajectory,
        step_mapping.step.Episode: custom_mapping.onto.Checkin,
    }

    # List of classes for which we have to build a mapper based on the instance's class name
    mapper_builder_fns = {
        step_mapping.step.ContextualElement: lambda class_name: custom_mapping._get_or_create_class(
            class_name, base_classes=(step_mapping.step.ContextualElement,)
        ),
        step_mapping.step.FeatureOfInterest: lambda class_name: lambda instance: "has"
        + class_name[4:],  # FOI_X
        step_mapping.step.QualitativeDescription: lambda class_name: custom_mapping._get_or_create_class(
            class_name, base_classes=(step_mapping.step.QualitativeDescription,)
        ),
    }

    @timed(logger.info, "'step_mapping_to_custom_mapping' warm up done in {}s.")
    def warm_up():
        # Searching for these entities at the beginning seems to improve performance.
        # This makes sense considering that owlready2 will cache this info (somehow).
        # TODO we should probably benchmark this
        step_mapping.onto.search(is_a=step_mapping.step.Trajectory)[1:]
        step_mapping.onto.search(is_a=step_mapping.step.FeatureOfInterest)[1:]
        step_mapping.onto.search(is_a=step_mapping.step.Episode)[1:]
        step_mapping.onto.search(is_a=step_mapping.step.ContextualElement)[1:]

    # begin at each agent
    with custom_mapping.onto:
        warm_up()

        # first entry is simply the class, not an instance
        agents = step_mapping.onto.search(is_a=step_mapping.step.Agent)[1:]

        for agent in tqdm(agents, desc="agents"):
            process_agent(agent)

        custom_mapping.add_entity2entity_relations(entity2entity_relations)

        # let's process the custom relations that were added when building the STEP representation
        for relation, domain, range in tqdm(
            step_mapping._custom_relations, desc="custom relations"
        ):
            domain = map_step_to_custom_entity(domain)
            range = map_step_to_custom_entity(range)
            custom_relations.add_to_relation(relation, domain, range)
        custom_mapping.add_entity2entity_relations(custom_relations)

    return custom_mapping
