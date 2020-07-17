import argparse
from pathlib import Path
import os, sys, inspect

import logging

from stlarm import data, onto


logging.basicConfig()
logger = logging.getLogger("stlarm")
logger.setLevel(logging.INFO)

FILTERED_NT = "resources/{}_filtered_triples.nt"
ALL_NT = "resources/{}_all_triples.nt"


def create_dirs():
    Path("cache").mkdir(parents=True, exist_ok=True)
    Path("resources").mkdir(parents=True, exist_ok=True)


def save_all(onto):
    filtered_nt = FILTERED_NT.format(onto.pretty_id)
    all_nt = ALL_NT.format(onto.pretty_id)
    onto.save(folder="cache")
    onto.save_filtered_triples(filtered_nt)
    onto.save_triples(all_nt)
    logger.info(f"Saved {str(type(onto))} at '{all_nt}' and '{filtered_nt}'.")


def get_available_data_sources():
    # Get the classes in stlarm.data which are concrete subclasses of data.TrajData
    classes = inspect.getmembers(
        data,  # data is the stlarm.data module
        lambda x: inspect.isclass(x)
        and not inspect.isabstract(x)
        and issubclass(x, data.TrajData),
    )
    return [c[0] for c in classes]


def main(
    name,
    working_dir,
    data_source_cls,
    time_granularity,
    max_hours_diff,
    max_distance_km,
    excluded_relations,
    dry_run,
    verbose,
):
    if verbose:
        logger.setLevel(logging.DEBUG)

    STEP_ONTO_NAME = f"step_{name}"
    CONVERTED_ONTO_NAME = f"converted_{name}"

    # change working dir
    os.chdir(working_dir)

    # Create dirs if necessary
    create_dirs()

    # Instantiate data source class
    data_src = getattr(data, data_source_cls)()

    # TODO Allow specifying uids/tids
    data_df, trajs, traj_data = data_src.get_all_data()
    logger.info(f"{data_df['uid'].nunique()} users")
    logger.info(f"{data_df['tid'].nunique()} trajectories")
    logger.info(f"{data_df['cid'].nunique()} checkins")
    logger.info(f"{data_df['venue_id'].nunique()} venues")

    step = onto.STEPRepresentation(
        name=STEP_ONTO_NAME,
        time_granularity=time_granularity,
        max_hours_diff=max_hours_diff,
        max_distance_km=max_distance_km,
        excluded_relations=excluded_relations,
    )
    step.gen_representation(trajs, traj_data=traj_data)
    logger.info("STEP has %d individuals", len(list(step.onto.individuals())))

    if not dry_run:
        save_all(step)

    converted = step.to_custom_representation(name=CONVERTED_ONTO_NAME)
    logger.info("CONVERTED has %d individuals", len(list(converted.onto.individuals())))

    if not dry_run:
        save_all(converted)

    if dry_run:
        logger.warning(
            "Executed on dry run mode. Everything was executed, except saving to disk."
        )

    logger.info("Finished.")


# TODO Implement load STEP and then build only Custom
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate onto data.")

    parser.add_argument(
        "-d",
        "--data-source-cls",
        action="store",
        type=str,
        metavar="CLASS_NAME",
        required=True,
        choices=get_available_data_sources(),
        help="name of class from which load data",
    )

    parser.add_argument(
        "-n",
        "--name",
        action="store",
        type=str,
        metavar="NAME",
        required=True,
        help="name of the ontology (which will generate step_NAME and converted_NAME.",
    )

    parser.add_argument(
        "-wd",
        "--working-dir",
        action="store",
        type=str,
        metavar="PATH",
        required=True,
        help="path to use as current working directory.",
    )

    parser.add_argument(
        "--time-granularity",
        action="store",
        type=str,
        metavar="GRANULARITY",
        required=True,
        choices=onto.OntoMapping.TIME_GRANULARITIES,
        help=f"time granularity.",
    )

    parser.add_argument(
        "--max-hours-diff",
        action="store",
        type=int,
        metavar="HOURS",
        required=True,
        help="max hours diff threshold for withinTimeWindow.",
    )

    parser.add_argument(
        "--max-distance-km",
        action="store",
        type=int,
        metavar="KMs",
        required=True,
        help="max distance threshold in kilometers for withinRadius.",
    )

    parser.add_argument(
        "--excluded-relations",
        type=str,
        nargs="*",
        default=[],
        choices=onto.OntoRepresentation.AVAILABLE_CUSTOM_RELATIONS,
        help="relations to exclude.",
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="process, but do not save.",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="increase output verbosity.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logger.info("CLI called as %s", sys.argv)
    args = vars(parse_arguments())
    main(**args)
