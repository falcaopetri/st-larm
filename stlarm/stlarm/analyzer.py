import io
import logging
import os
from pathlib import Path
import tempfile
import subprocess

import pandas as pd
import numpy as np

from .rule import get_str_triples, is_constant
from .utils import get_system_memory_info, timed


AMIE_JAR_PATH = os.environ["AMIE_JAR_PATH"]
logger = logging.getLogger("stlarm.analyzer")
logger.debug("Using AMIE_JAR_PATH %s", AMIE_JAR_PATH)


def execute_java_process(classpath, args, output_file, max_heap_size_gib=None):
    if max_heap_size_gib:
        free_memory = get_system_memory_info()["free"]
        if free_memory <= max_heap_size_gib:
            raise ValueError(
                f"Required max heap size is {max_heap_size_gib}G, but only {free_memory} seems to be available."
            )
        command = [
            "java",
            f"-Xmx{max_heap_size_gib}G",
            "-cp",
            str(AMIE_JAR_PATH),
            classpath,
        ]
    else:
        command = ["java", "-cp", str(AMIE_JAR_PATH), classpath]

    args = [str(x) for x in args]
    logger.debug("Calling %s", " ".join(command + args))
    try:
        stdout = subprocess.check_output(command + args, stderr=subprocess.PIPE).decode(
            "utf-8"
        )
    except subprocess.CalledProcessError as e:
        logger.error("Java process returned non-zero exit code")
        logger.error("STDOUT: %s", e.stdout)
        logger.error("STDERR: %s", e.stderr)
        raise e

    if output_file:
        with open(output_file, "w") as fout:
            fout.write(stdout)
        logger.info(f"Saved java process output to {output_file}.")

    logger.info(f"[Done] {classpath}")
    return stdout


def get_skiprows(file_or_str):
    """Returns the index for those lines which do not contain \t"""
    try:
        with open(file_or_str, "r") as f:
            lines = f.readlines()
    except:
        lines = file_or_str.split("\n")

    for idx, l in enumerate(lines):
        if "\t" not in l:
            yield idx


def has_unsorted_reflexive_relation(triples):
    for s, p, o in triples:
        # is reflexive relation
        if p in ["<withinTimeWindow>", "<withinRadius>"]:
            if s > o:
                # unsorted
                return True
    return False


class KBAnalyzer:
    def _get_kb_data(self, kb_path, output_file, max_heap_size_gib):
        class_name = "amie.data.utils.KBsSummarizer"
        args = [kb_path]
        train_out = execute_java_process(
            class_name, args, output_file, max_heap_size_gib=max_heap_size_gib,
        )
        logger.debug("Raw kb summary: %s", train_out)
        kb_df = pd.read_csv(
            io.StringIO(train_out), sep="\t", skiprows=get_skiprows(train_out),
        ).sort_values("Triples", ascending=False)

        raw_df = pd.read_csv(kb_path, sep="\t", names="s p o .".split())
        return kb_df.append(
            {
                "Relation": "TOTAL",
                "Triples": len(raw_df),
                "Functionality": np.nan,
                "Inverse functionality": np.nan,
                "Variance": np.nan,
                "Inverse Variance": np.nan,
                "Number of subjects": raw_df["s"].nunique(),
                "Number of objects": raw_df["o"].nunique(),
            },
            ignore_index=True,
        )

    def analyse_kb(self, kb_file, output_file=None, max_heap_size_gib=5):
        logger.info(f"Processing {kb_file}")
        kb_df = self._get_kb_data(kb_file, output_file, max_heap_size_gib)
        return kb_df


class RulesAnalyzer:
    def __init__(
        self,
        rules_file,
        train_kb=None,
        target_kb=None,
        metarules_class_file=None,
        remove_reflexive_duplicates=True,
        metarules=False,
    ):
        self.rules_file = rules_file
        self.train_kb = train_kb
        self.target_kb = target_kb
        self.metarules_class_file = metarules_class_file
        self._raw_df = RulesAnalyzer.get_raw_df(
            self.rules_file, remove_reflexive_duplicates,
        )
        self.df = RulesAnalyzer.enhance_df(self._raw_df, metarules)

        if self.metarules_class_file:
            self.metarules_class_df = pd.read_csv(self.metarules_class_file)
            self.df = self.df.merge(self.metarules_class_df, on="Metarule")

    @classmethod
    def from_analyzer(
        cls,
        other: "RulesAnalyzer",
        df: pd.DataFrame,
        metarules_class_file=None,
        metarules=False,
    ):
        # TODO could have a better interface
        with tempfile.NamedTemporaryFile() as temp_data_file:
            df.to_csv(temp_data_file.name, sep="\t", index=False)
            return cls(
                temp_data_file.name,
                other.train_kb,
                other.target_kb,
                metarules_class_file=metarules_class_file,
                remove_reflexive_duplicates=False,
                metarules=metarules,
            )

    @staticmethod
    def get_raw_df(rules_file, remove_reflexive_duplicates):
        df = pd.read_csv(rules_file, sep="\t", skiprows=get_skiprows(rules_file))

        logger.info(f"Loaded {len(df)} rules.")
        if remove_reflexive_duplicates:
            logger.info("Improving rules...")
            logger.info("Removing rules with unsorted reflexive relations...")
            _prev_size = len(df)
            df = df[
                ~df["Rule"]
                .apply(get_str_triples)
                .apply(has_unsorted_reflexive_relation)
            ].copy()
            logger.info(
                f"Removed {_prev_size-len(df)} rules with unsorted reflexive relations. {len(df)} rules will be used."
            )

        return df

    @staticmethod
    def enhance_df(df, metarules):
        df = df.copy()
        if metarules:
            metarules_df = RulesAnalyzer._execute_metarule_builder(df)
            df = df.merge(metarules_df, on="Rule")
            logger.info(
                f"Added 'Metarule' column. There are {df['Metarule'].nunique()} metarules."
            )

        df["Rule"] = df["Rule"].replace("!=", "   <DIFFERENT>  ", regex=True)
        df = (
            df["Rule"]
            .str.split("=>", expand=True)
            .rename(columns={0: "body", 1: "head"})
            .join(df)
            .sort_values(["Std Confidence", "Head Coverage"], ascending=False)
        )
        df["head relation"] = df["head"].str.split("  ", expand=True)[1]
        df["length"] = df["Rule"].apply(get_str_triples).apply(len)
        return df

    @staticmethod
    @timed(
        logger.debug, "Building meta-rules lattice using amie.MetaruleBuilder took {}s."
    )
    def _execute_metarule_builder(df):
        with tempfile.NamedTemporaryFile() as temp_data_file:
            df.to_csv(temp_data_file.name, sep="\t", decimal=",", index=False)

            class_name = "amie.rules.eval.MetaruleBuilder"
            # file with rules data
            args = [temp_data_file.name]
            metarules_txt = execute_java_process(class_name, args, None)

        return pd.read_csv(
            io.StringIO(metarules_txt), sep="\t", names=["Rule", "Metarule"]
        )

    def analyse_head_predicates(self):
        return self.df["head relation"].value_counts()

    def analyse_body_predicates_per_head_predicate(self):
        # TODO use Rule.get_str_triples
        body = np.array(
            self.df["body"]
            .str.strip()
            .str.split("  ")
            .apply(lambda x: np.array(x).reshape((-1, 3)))
            .values
        )

        body_predicates = [x.T[1] for x in body]
        head_predicate = self.df["head relation"].values

        predicates_df = (
            pd.DataFrame({"head": head_predicate, "body": body_predicates})
            .explode("body")
            .astype(str)
        )

        body_counts = predicates_df.groupby(["head", "body"]).size().unstack()
        ax = body_counts.plot.barh(figsize=(10, 5))
        ax.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))
        return ax

    def _num_X_per_Z(self, X, Z, thresholds, **plot_kwargs):
        X_per_Z = []
        for threshold in thresholds:
            count = self.df[(self.df[Z].astype(float) >= threshold)][X].nunique()
            X_per_Z.append((threshold, count))

        ax = pd.DataFrame(X_per_Z, columns=["threshold", X]).plot.line(
            x="threshold", y=X, marker="o", **plot_kwargs
        )
        plot_kwargs.pop("ax", "")
        ax.set_title(
            f"Num of {X} per {Z}" + (f" ({plot_kwargs})" if plot_kwargs else "")
        )

        ax.set_xlabel(f"{Z} threshold")
        ax.set_ylabel(f"Num {X}")
        ax.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))
        return ax

    def _num_X_per_confidence(self, X, conf, **plot_kwargs):
        return self._num_X_per_Z(X, conf, np.linspace(0, 1, num=11), **plot_kwargs)

    def num_rules_per_confidence(self, conf="Std Confidence", **plot_kwargs):
        return self._num_X_per_confidence("Rule", conf, **plot_kwargs)

    def num_metarules_per_confidence(self, conf="Std Confidence", **plot_kwargs):
        return self._num_X_per_confidence("Metarule", conf, **plot_kwargs)

    def _num_X_per_support(self, X, step, **plot_kwargs):
        max_support = self.df["Positive Examples"].max()
        return self._num_X_per_Z(
            X,
            "Positive Examples",
            np.arange(0, max_support, step=step),
            logy=True,
            **plot_kwargs,
        )

    def num_rules_per_support(self, step=100, **plot_kwargs):
        return self._num_X_per_support("Rule", step, **plot_kwargs)

    def num_metarules_per_support(self, step=100, **plot_kwargs):
        return self._num_X_per_support("Metarule", step, **plot_kwargs)

    def _num_X_per_Z_foreach_Y(self, X, Y, Z, thresholds, **plot_kwargs):
        Y_per_Z = []
        for threshold in thresholds:
            filtered = self.df[(self.df[Z].astype(float) >= threshold)]
            counts = filtered[[X, Y]].drop_duplicates()[Y].value_counts()
            counts = counts.rename(threshold)
            Y_per_Z.append(counts)

        ax = (
            pd.concat(Y_per_Z, axis=1)
            .sort_index()
            .T.plot.line(marker="o", **plot_kwargs)
        )
        plot_kwargs.pop("ax", "")
        ax.set_title(
            f"Num of {X} per {Z} for each {Y}"
            + (f" ({plot_kwargs})" if plot_kwargs else "")
        )
        ax.set_xlabel(f"{Z} threshold")
        ax.set_ylabel(f"Num {X}")
        ax.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))
        return ax

    def _num_X_per_confidence_foreach_Y(self, X, Y, conf, **plot_kwargs):
        return self._num_X_per_Z_foreach_Y(
            X, Y, conf, np.linspace(0, 1, num=11), **plot_kwargs
        )

    def num_rules_per_confidence_foreach_head(
        self, conf="Std Confidence", **plot_kwargs
    ):
        return self._num_X_per_confidence_foreach_Y(
            "Rule", "head relation", conf, **plot_kwargs
        )

    def num_metarules_per_confidence_foreach_head(
        self, conf="Std Confidence", **plot_kwargs
    ):
        return self._num_X_per_confidence_foreach_Y(
            "Metarule", "head relation", conf, **plot_kwargs
        )

    def num_rules_per_confidence_foreach_label(
        self, conf="Std Confidence", **plot_kwargs
    ):
        return self._num_X_per_confidence_foreach_Y(
            "Rule", "classification", conf, **plot_kwargs
        )

    def num_metarules_per_confidence_foreach_label(
        self, conf="Std Confidence", **plot_kwargs
    ):
        return self._num_X_per_confidence_foreach_Y(
            "Metarule", "classification", conf, **plot_kwargs
        )

    def _num_X_per_support_foreach_Y(self, X, Y, step, **plot_kwargs):
        max_support = self.df["Positive Examples"].max()
        return self._num_X_per_Z_foreach_Y(
            X,
            Y,
            "Positive Examples",
            np.arange(0, max_support, step=step),
            **plot_kwargs,
        )

    def num_rules_per_support_foreach_head(self, step=100, **plot_kwargs):
        return self._num_X_per_support_foreach_Y(
            "Rule", "head relation", step, **plot_kwargs
        )

    def num_metarules_per_support_foreach_head(self, step=100, **plot_kwargs):
        return self._num_X_per_support_foreach_Y(
            "Metarule", "head relation", step, **plot_kwargs
        )

    def num_rules_per_support_foreach_label(self, step=100, **plot_kwargs):
        return self._num_X_per_support_foreach_Y(
            "Rule", "classification", step, **plot_kwargs
        )

    def num_metarules_per_support_foreach_label(self, step=100, **plot_kwargs):
        return self._num_X_per_support_foreach_Y(
            "Metarule", "classification", step, **plot_kwargs
        )
