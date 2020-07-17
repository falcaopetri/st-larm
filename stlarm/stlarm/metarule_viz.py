from termcolor import colored

from .rule import get_str_triples, is_constant


def to_datalog_triple(triple, should_colorize, highlights, highlights_color):
    triple = [x.replace("<", "").replace(">", "") for x in triple]
    s, p, o = triple

    def colorize_entity(x):
        if x in highlights:
            x = colored(x, on_color=highlights_color)
        elif x.startswith("?z"):
            x = colored(x, on_color="on_green")
        elif is_constant(x):
            x = colored(x, on_color="on_red")
        return x

    if should_colorize:
        s = colorize_entity(s)
        if p in highlights:
            p = colored(p, on_color=highlights_color)
        o = colorize_entity(o)

    return f"{p}({s}, {o})"


def to_datalog_rule(
    rule, should_colorize=True, highlights=set(), highlights_color="on_yellow"
):
    body, head = get_str_triples(rule, "both")
    head = to_datalog_triple(head, should_colorize, highlights, highlights_color)

    if body is not None:
        body = ", ".join(
            [
                to_datalog_triple(triple, should_colorize, highlights, highlights_color)
                for triple in body
            ]
        )
    else:
        body = ""
    return f"{body} => {head}"


def print_metarule(
    df, metarule, highlight_entities=set(), n_examples=5, only_highlighted=False
):
    highlight_entities = set(highlight_entities)

    metarule_data = df[df["Metarule"] == metarule].sort_values("Std Confidence")
    hc = metarule_data["Head Coverage"].sum()
    conf_min = metarule_data["Std Confidence"].min()
    conf_max = metarule_data["Std Confidence"].max()

    print(
        to_datalog_rule(metarule),
        colored(
            f"(hc: {hc:.4f}, min-std-conf: {conf_min:.2f}, max-std-conf: {conf_max:.2f})",
            "blue",
        ),
    )

    is_highlight = metarule_data["Rule"].str.contains("|".join(highlight_entities))
    highlight_rules = metarule_data[is_highlight].head(n_examples)
    non_highlight_rules = metarule_data[~is_highlight].head(n_examples)

    if is_highlight.any():
        for rule, hc, conf in highlight_rules[
            ["Rule", "Head Coverage", "Std Confidence"]
        ].itertuples(index=False, name=None):
            print(
                "\t",
                to_datalog_rule(rule, highlights=highlight_entities),
                colored(f"(hc: {hc:.4f}, std-conf: {conf:.2f})", "blue"),
            )

    if is_highlight.any() and not is_highlight.all() and not only_highlighted:
        print("-" * 10)

    if not is_highlight.all() and not only_highlighted:
        for rule, hc, conf in non_highlight_rules[
            ["Rule", "Head Coverage", "Std Confidence"]
        ].itertuples(index=False, name=None):
            print(
                "\t",
                to_datalog_rule(rule),
                colored(f"(hc: {hc:.4f}, std-conf: {conf:.2f})", "blue"),
            )

    if len(metarule_data) > n_examples:
        print(f"\t... (total: {len(metarule_data)})")
