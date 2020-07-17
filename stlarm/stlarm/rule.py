import numpy as np


def is_constant(rule):
    return not rule.startswith("?")


def get_constants(rule):
    triples = rule.get_str_triples()
    # TODO Make this more efficient
    # Constants should be added in the order they appear in the rule
    constants = list()
    for s, _, o in triples:
        if is_constant(s) and s not in constants:
            constants.append(s)
        if is_constant(o) and o not in constants:
            constants.append(o)
    return constants


def has_constants(rule):
    return len(get_constants(rule)) == 0


def get_str_triples(rule, ret="all"):
    assert ret in ["all", "both", "body", "head"]

    def to_triples(x):
        x = x.strip()
        if len(x) == 0:
            return None

        x = x.split()
        return np.array(x).reshape((-1, 3))

    rule_str = str(rule)
    body, head = rule_str.split(" => ")
    body = to_triples(body)
    head = to_triples(head)

    if ret == "all":
        return np.append(body, head, axis=0)
    elif ret == "both":
        return body, head.squeeze()
    elif ret == "body":
        return body
    elif ret == "head":
        return head.squeeze()
