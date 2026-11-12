"""Portable file paths in saved patches.

A patch written to disk carries no machine-specific prefix: wherever a string
in it starts with a known root, the root is replaced by a token, and the token
is expanded again when the patch is read. Nodes only ever see ordinary absolute
paths; the tokens exist only in the file.

The roots, longest first so the most specific one wins:
    $DPG_HOME     the folder holding the dpg_system package (the repo root)
    $<NAME>       each entry of "path_roots" in dpg_system_config.json, e.g.
                  "path_roots": {"BMO_LAB": "/Users/me/Projects/BMO_Lab"}
    $HOME         the home folder

A path is matched where it begins a string or follows whitespace or a quote,
so a message box holding 'open /Users/me/x.wav' is converted too. Root names
are capital letters, so text such as '$x' in an expression is never touched.
A token whose root is not defined on this machine is left as it is, with a
note in the console. A '~' already in a patch is never touched either way.
Each root replaces its prefix only when followed by a path separator, the end
of the string, whitespace or a quote, so /a/b never matches inside /a/bc.
"""

import os
import re
from pathlib import Path

DPG_HOME = str(Path(__file__).resolve().parent.parent)

_user_roots = {}
_warned_missing = set()
_contract_rules = None
_expand_pattern = None


def configure(path_roots):
    global _user_roots, _contract_rules, _expand_pattern
    _user_roots = {}
    if isinstance(path_roots, dict):
        for name, root in path_roots.items():
            if isinstance(name, str) and isinstance(root, str) and re.fullmatch(r'[A-Z_][A-Z0-9_]*', name):
                _user_roots[name] = os.path.expanduser(root).rstrip('/\\')
            else:
                print(f'path_roots: ignoring {name!r}: {root!r} (names are capital letters, digits and _)')
    _contract_rules = None
    _expand_pattern = None


def _roots():
    roots = {'HOME': os.path.expanduser('~').rstrip('/\\'), 'DPG_HOME': DPG_HOME}
    roots.update(_user_roots)
    return roots


# Matched only where a path can begin: start of string, or after whitespace or a quote.
_BOUNDARY = r'(?<![^\s\'"])'


def _rules():
    global _contract_rules
    if _contract_rules is None:
        entries = [(root, '$' + name) for name, root in _roots().items() if root]
        entries.sort(key=lambda e: len(e[0]), reverse=True)
        _contract_rules = [(re.compile(_BOUNDARY + re.escape(root) + r'(?=[/\\]|$|[\s\'"])'), token)
                           for root, token in entries]
    return _contract_rules


def contract_string(s):
    if '/' not in s and '\\' not in s:
        return s
    for pattern, token in _rules():
        s = pattern.sub(lambda m, t=token: t, s)
    return s


def _expand_match(m):
    name = m.group(1)
    root = _roots().get(name)
    if root is None:
        if name not in _warned_missing:
            _warned_missing.add(name)
            print(f'portable paths: ${name} is not defined here - add it to "path_roots" in dpg_system_config.json')
        return m.group(0)
    return root


def expand_string(s):
    global _expand_pattern
    if '$' not in s:
        return s
    if _expand_pattern is None:
        _expand_pattern = re.compile(_BOUNDARY + r'\$([A-Z_][A-Z0-9_]*)(?=[/\\]|$|[\s\'"])')
    return _expand_pattern.sub(_expand_match, s)


def _walk(value, convert):
    if isinstance(value, str):
        return convert(value)
    if isinstance(value, dict):
        return {k: _walk(v, convert) for k, v in value.items()}
    if isinstance(value, list):
        return [_walk(v, convert) for v in value]
    if isinstance(value, tuple):
        return tuple(_walk(v, convert) for v in value)
    return value


def contract(container):
    """A copy of a patch container with known roots replaced by tokens."""
    return _walk(container, contract_string)


def expand(container):
    """A copy of a patch container with tokens replaced by this machine's roots."""
    return _walk(container, expand_string)
