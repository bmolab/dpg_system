"""
Converter utilities for OSCQuery JSON → dpg_system node specs.

Provides functions to:
- Convert an OSCQuery parameter spec to a dpg_system node specification
- Walk an OSCQuery tree and yield node specs for all leaf parameters
- Search a tree for parameters matching a query
- Detect repeating indexed (channel) patterns in the tree
"""

import re


def param_to_node_spec(osc_spec):
    """
    Convert an OSCQuery parameter dict to a dpg_system node specification.

    Args:
        osc_spec: dict with TYPE, VALUE, RANGE, ACCESS, DESCRIPTION, FULL_PATH, etc.

    Returns:
        dict with keys: node_type, args, value, min, max, choices, description, access
        Returns None if the spec cannot be converted.
    """
    if not isinstance(osc_spec, dict) or 'TYPE' not in osc_spec:
        return None

    types = osc_spec['TYPE']
    value = osc_spec.get('VALUE', [])
    access = osc_spec.get('ACCESS', 3)
    description = osc_spec.get('DESCRIPTION', '')
    full_path = osc_spec.get('FULL_PATH', '')

    spec = {
        'value': value,
        'min': None,
        'max': None,
        'choices': None,
        'description': description,
        'access': access,
        'full_path': full_path,
        'node_type': None,
        'args': [],
    }

    # Extract range info
    ranges = osc_spec.get('RANGE', [])

    if types == 'f':
        # Single float
        spec['node_type'] = 'osc_slider'
        if len(ranges) > 0:
            spec['min'] = ranges[0].get('MIN', 0.0)
            spec['max'] = ranges[0].get('MAX', 1.0)
        if len(value) > 0:
            spec['value'] = value[0]

    elif types == 'i':
        # Single integer
        spec['node_type'] = 'osc_int'
        if len(ranges) > 0:
            spec['min'] = ranges[0].get('MIN', 0)
            spec['max'] = ranges[0].get('MAX', 100)
        if len(value) > 0:
            spec['value'] = value[0]

    elif types == 's':
        # String — check for VALS (menu choices)
        vals = None
        if len(ranges) > 0 and 'VALS' in ranges[0]:
            vals = ranges[0]['VALS']
        elif 'VALS' in osc_spec:
            vals = osc_spec['VALS']

        if vals and len(vals) > 0:
            spec['node_type'] = 'osc_menu'
            spec['choices'] = vals
        else:
            spec['node_type'] = 'osc_string'

        if len(value) > 0:
            spec['value'] = value[0]

    elif types in ('F', 'T'):
        # Boolean / toggle
        spec['node_type'] = 'osc_toggle'
        if len(value) > 0:
            spec['value'] = value[0]

    elif types == 'N':
        # Null / impulse / button
        spec['node_type'] = 'osc_button'

    elif types == 'r':
        # Color — treat as string for now
        spec['node_type'] = 'osc_string'
        if len(value) > 0:
            spec['value'] = value[0]

    elif len(types) > 1 and all(c == 'f' for c in types):
        # Multiple floats (ff, fff, ffff) — create multiple sliders or a vector
        count = len(types)
        spec['node_type'] = 'osc_float'
        spec['value'] = value
        if len(ranges) >= count:
            spec['min'] = [r.get('MIN', 0.0) for r in ranges]
            spec['max'] = [r.get('MAX', 1.0) for r in ranges]
        spec['args'] = [str(count)]  # pass count as arg

    elif len(types) > 1 and all(c == 'i' for c in types):
        # Multiple ints
        count = len(types)
        spec['node_type'] = 'osc_int'
        spec['value'] = value
        if len(ranges) >= count:
            spec['min'] = [r.get('MIN', 0) for r in ranges]
            spec['max'] = [r.get('MAX', 100) for r in ranges]
        spec['args'] = [str(count)]

    else:
        # Unknown type — fall back to message/string
        spec['node_type'] = 'osc_message'
        if len(value) > 0:
            spec['value'] = value

    return spec


def walk_tree(json_tree, path_prefix=''):
    """
    Walk an OSCQuery tree and yield (path, node_spec) for all leaf parameters.

    Args:
        json_tree: dict with CONTENTS, TYPE, etc.
        path_prefix: current path prefix

    Yields:
        (full_osc_path, node_spec_dict) tuples
    """
    if not isinstance(json_tree, dict):
        return

    # If this node has a TYPE, it's a leaf parameter
    if 'TYPE' in json_tree:
        spec = param_to_node_spec(json_tree)
        if spec:
            yield (path_prefix, spec)

    # Recurse into CONTENTS
    if 'CONTENTS' in json_tree and isinstance(json_tree['CONTENTS'], dict):
        for key, child in json_tree['CONTENTS'].items():
            child_path = path_prefix + '/' + key if path_prefix else '/' + key
            yield from walk_tree(child, child_path)


def search_tree(json_tree, query, current_path='', case_insensitive=True):
    """
    Search for parameters matching a query string in the tree.

    Returns list of (path, param_dict) tuples.
    """
    results = []
    if not isinstance(json_tree, dict):
        return results

    query_cmp = query.lower() if case_insensitive else query

    if 'CONTENTS' in json_tree:
        for key, child in json_tree['CONTENTS'].items():
            child_path = current_path + '/' + key
            key_cmp = key.lower() if case_insensitive else key

            if query_cmp in key_cmp:
                if 'TYPE' in child:
                    results.append((child_path, child))
                else:
                    # Add all leaves under this matching container
                    for path, spec in walk_tree(child, child_path):
                        results.append((path, child.get('CONTENTS', {}).get(
                            path.split('/')[-1], child) if 'CONTENTS' in child else child))

            # Always recurse
            results.extend(search_tree(child, query, child_path, case_insensitive))

    return results


def detect_channel_pattern(json_tree, path=''):
    """
    Detect repeating indexed structures in the tree (like EOS channels).

    Returns a list of dicts:
    [
        {
            'path': '/chan',
            'indices': [1, 2, 3, ...],
            'template_path': '/chan/{N}/param',
            'params': [list of param names under each index]
        }
    ]
    """
    patterns = []
    if not isinstance(json_tree, dict):
        return patterns

    if 'CONTENTS' not in json_tree:
        return patterns

    contents = json_tree['CONTENTS']

    # Check if children are numbered (indexed)
    numbered_children = {}
    non_numbered = []
    for key in contents:
        if re.match(r'^\d+$', key):
            numbered_children[int(key)] = contents[key]
        else:
            non_numbered.append(key)

    if len(numbered_children) >= 2:
        # This looks like a channel pattern
        indices = sorted(numbered_children.keys())

        # Get the structure of the first index to use as template
        first = numbered_children[indices[0]]
        param_names = []
        if 'CONTENTS' in first:
            param_names = list(first['CONTENTS'].keys())

        patterns.append({
            'path': path,
            'indices': indices,
            'template_path': path + '/{N}',
            'params': param_names,
        })

    # Recurse into all children
    for key, child in contents.items():
        child_path = path + '/' + key
        patterns.extend(detect_channel_pattern(child, child_path))

    return patterns


def get_contents_list(json_tree, path=''):
    """
    Get the immediate contents of a path in the tree for browsing.

    Returns list of dicts: [{'name': key, 'is_container': bool, 'type': str, 'path': full_path}, ...]
    """
    node = navigate_to_path(json_tree, path)
    if node is None:
        return []

    results = []
    if 'CONTENTS' in node and isinstance(node['CONTENTS'], dict):
        for key, child in node['CONTENTS'].items():
            # Skip non-dict entries (metadata like ACCESS, TYPE at this level)
            if not isinstance(child, dict):
                continue

            # Determine TYPE — it could be a sibling of CONTENTS or inside it
            child_type = child.get('TYPE', '')
            if not child_type and 'CONTENTS' in child and isinstance(child['CONTENTS'], dict):
                child_type = child['CONTENTS'].get('TYPE', '')

            # A node is a true container only if its CONTENTS has actual child dicts
            # (not just metadata like TYPE, VALUE, RANGE, etc.)
            has_real_children = False
            if 'CONTENTS' in child and isinstance(child['CONTENTS'], dict):
                has_real_children = any(
                    isinstance(v, dict) for v in child['CONTENTS'].values()
                )

            entry = {
                'name': key,
                'path': path + '/' + key if path else '/' + key,
                'is_container': has_real_children,
                'type': child_type,
                'description': child.get('DESCRIPTION', key),
            }
            results.append(entry)

    return results


def navigate_to_path(json_tree, path):
    """Navigate a JSON tree to the node at the given OSC path."""
    if not path or path == '/':
        return json_tree

    components = [c for c in path.split('/') if c]
    current = json_tree
    for comp in components:
        if 'CONTENTS' in current and comp in current['CONTENTS']:
            current = current['CONTENTS'][comp]
        elif comp in current:
            current = current[comp]
        else:
            return None
    return current
