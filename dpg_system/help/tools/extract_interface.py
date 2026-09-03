"""Static extractor: for each registered node label, report the class that
builds it and the inputs / outputs / properties / options it declares.

Reads the source with ast -- no import, no GUI, no torch.
"""
import ast, os, sys, json, collections

ROOT = '/Users/drokeby/dpg_system/dpg_system'
SKIP = ('.claude', 'worktrees', 'gang_analysis', 'noise_estimation',
        'logodds_evaluator', 'smpl_utilities', 'magnetometer_assessment')

_IN = ['add_input', 'add_bool_input', 'add_float_input', 'add_int_input',
       'add_string_input', 'add_list_input', 'add_array_input', 'add_tensor_input',
       'add_dim_input', 'add_shape_input', 'add_word_input', 'add_signal_input',
       'add_trigger_signal_input', 'add_scaling_signal_input', 'add_modulation_input']
_OUT = ['add_output', 'add_bool_output', 'add_float_output', 'add_int_output',
        'add_string_output', 'add_list_output', 'add_array_output',
        'add_tensor_output', 'add_signal_output']
_OPT = ['add_option', 'add_dim_option', 'add_style_option', 'add_shading_option',
        'add_sample_count_option', 'add_min_and_max_x_options',
        'add_min_and_max_y_options']
ADDERS = {}
for _n in _IN:  ADDERS[_n] = 'input'
for _n in _OUT: ADDERS[_n] = 'output'
for _n in _OPT: ADDERS[_n] = 'option'
ADDERS['add_property'] = 'property'
ADDERS['add_switch'] = 'option'
ADDERS['add_display'] = 'display'
ADDERS['add_label'] = 'label'


def lit(node):
    """Best-effort literal for an ast node; None if not static."""
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def src_of(node, lines):
    try:
        return ast.get_source_segment('\n'.join(lines), node)
    except Exception:
        return None


class FileInfo:
    def __init__(self, path):
        self.path = path
        self.src = open(path, encoding='utf-8', errors='replace').read()
        self.lines = self.src.split('\n')
        self.tree = ast.parse(self.src)
        self.classes = {}        # name -> ast.ClassDef
        self.bases = {}          # name -> [base names]
        self.registrations = []  # (label, factory_class, data)
        for n in ast.walk(self.tree):
            if isinstance(n, ast.ClassDef):
                self.classes[n.name] = n
                self.bases[n.name] = [b.id for b in n.bases
                                      if isinstance(b, ast.Name)] + \
                                     [b.attr for b in n.bases
                                      if isinstance(b, ast.Attribute)]
            if isinstance(n, ast.Call) and getattr(n.func, 'attr', '') == 'register_node':
                if not n.args:
                    continue
                label = lit(n.args[0])
                if not isinstance(label, str):
                    continue
                cls = None
                if len(n.args) > 1:
                    f = n.args[1]
                    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                        cls = f.value.id      # SomeNode.factory
                data = lit(n.args[2]) if len(n.args) > 2 else None
                self.registrations.append((label, cls, data))



def dispatch_map(cls_name, files):
    """If a class's factory() picks a subclass by name, map the names it tests
    to the class it returns.  Returns (mapping, default_class)."""
    for fi in files.values():
        cd = fi.classes.get(cls_name)
        if cd is None:
            continue
        fac = None
        for item in cd.body:
            if isinstance(item, ast.FunctionDef) and item.name == 'factory':
                fac = item
        if fac is None:
            return {}, None
        mapping, default = {}, None

        def ret_class(body):
            for st in body:
                if isinstance(st, ast.Return) and isinstance(st.value, ast.Call) \
                        and isinstance(st.value.func, ast.Name):
                    return st.value.func.id
            return None

        node = fac
        for st in ast.walk(fac):
            if not isinstance(st, ast.If):
                continue
            c = ret_class(st.body)
            if c is None:
                continue
            test = st.test
            keys = []
            if isinstance(test, ast.Compare) and isinstance(test.ops[0], ast.In):
                v = lit(test.comparators[0])
                if isinstance(v, list):
                    keys = [x for x in v if isinstance(x, str)]
            elif isinstance(test, ast.Compare) and isinstance(test.ops[0], ast.Eq):
                v = lit(test.comparators[0])
                if isinstance(v, str):
                    keys = [v]
            for k in keys:
                mapping[k] = c
            if st.orelse and not any(isinstance(x, ast.If) for x in st.orelse):
                d = ret_class(st.orelse)
                if d:
                    default = d
        return mapping, default
    return {}, None


def load_all():
    files = {}
    for root, dirs, names in os.walk(ROOT):
        if any(s in root for s in SKIP):
            continue
        for nm in names:
            if not nm.endswith('.py'):
                continue
            p = os.path.join(root, nm)
            try:
                files[p] = FileInfo(p)
            except SyntaxError:
                pass
    return files


def collect(cls_name, files, seen=None, depth=0, prefer=None):
    """Walk a class and its bases, gathering declared interface elements."""
    if seen is None:
        seen = set()
    if cls_name in seen or depth > 6:
        return []
    seen.add(cls_name)
    out = []
    # Class names repeat across modules (StringNode exists in three files), so
    # search the module that registered the label before falling back to any.
    ordered = list(files.values())
    if prefer is not None:
        ordered.sort(key=lambda f: 0 if f.path == prefer else 1)
    for fi in ordered:
        if cls_name not in fi.classes:
            continue
        cd = fi.classes[cls_name]
        # bases first, so inherited elements come first (matching construction order)
        for b in fi.bases.get(cls_name, []):
            out += collect(b, files, seen, depth + 1, prefer=fi.path)
        for n in ast.walk(cd):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)):
                continue
            kind = ADDERS.get(n.func.attr)
            if kind is None:
                continue
            label = lit(n.args[0]) if n.args else None
            if label is None:
                for kw in n.keywords:
                    if kw.arg == 'label':
                        label = lit(kw.value)
            kws = {kw.arg: lit(kw.value) for kw in n.keywords if kw.arg}
            entry_src = None
            if not isinstance(label, str):
                entry_src = src_of(n, fi.lines)
            out.append({
                'kind': kind,
                'label': label if isinstance(label, str) else '<dynamic>',
                'src': entry_src,
                'adder': n.func.attr,
                'widget_type': kws.get('widget_type'),
                'default': kws.get('default_value'),
                'triggers': bool(kws.get('triggers_execution')),
                'file': os.path.basename(fi.path),
                'line': n.lineno,
                'class': cls_name,
            })
        # help_file_name is often set on a BASE class (TorchDistributionNode
        # carries 't.dist_help' for every t.dist.* node), so record it here and
        # let collect()'s base walk carry it down to the subclasses.
        for n in ast.walk(cd):
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Attribute) \
                    and n.targets[0].attr == 'help_file_name':
                v = lit(n.value)
                if isinstance(v, str) and v:
                    out.append({'kind': 'help_file_name', 'label': v,
                                'widget_type': None, 'default': None,
                                'triggers': False, 'src': None,
                                'adder': 'help_file_name',
                                'file': os.path.basename(fi.path), 'line': n.lineno,
                                'class': cls_name})

        # name_archive entries: link resolution accepts these as alternative
        # port names, so a cord aimed at an archived name does connect.
        for n in ast.walk(cd):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                    and n.func.attr == 'append' and n.args \
                    and isinstance(n.func.value, ast.Attribute) \
                    and n.func.value.attr == 'name_archive':
                v = lit(n.args[0])
                if isinstance(v, str) and v:
                    out.append({'kind': 'alias', 'label': v, 'widget_type': None,
                                'default': None, 'triggers': False, 'src': None,
                                'adder': 'name_archive',
                                'file': os.path.basename(fi.path), 'line': n.lineno,
                                'class': cls_name})

        # renames: some nodes relabel a port after construction
        for n in ast.walk(cd):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                    and n.func.attr in ('set_label', 'set_name') and n.args:
                v = lit(n.args[0])
                if isinstance(v, str) and v:
                    out.append({'kind': 'alias', 'label': v, 'widget_type': None,
                                'default': None, 'triggers': False, 'src': None,
                                'adder': n.func.attr,
                                'file': os.path.basename(fi.path), 'line': n.lineno,
                                'class': cls_name})
        # combo item lists: <thing>.widget.combo_items = [...]
        for n in ast.walk(cd):
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Attribute) \
                    and n.targets[0].attr in ('combo_items', 'choices'):
                v = lit(n.value)
                if isinstance(v, list):
                    out.append({'kind': 'choices', 'label': src_of(n.targets[0], fi.lines) or '',
                                'widget_type': None, 'default': v, 'triggers': False,
                                'src': None, 'adder': 'combo_items',
                                'file': os.path.basename(fi.path), 'line': n.lineno,
                                'class': cls_name})
        break
    return out


def main():
    files = load_all()
    index = {}
    for fi in files.values():
        for label, cls, data in fi.registrations:
            if cls is None:
                continue
            dmap, ddef = dispatch_map(cls, files)
            if dmap:
                base = label.split('_')[-1]
                cls = dmap.get(base) or dmap.get(label) or ddef or cls
            index[label] = {
                'class': cls,
                'file': os.path.basename(fi.path),
                'data': data,
                'elements': collect(cls, files, prefer=fi.path),
            }
    json.dump(index, open(sys.argv[1] if len(sys.argv) > 1 else 'iface.json', 'w'), indent=1)
    print('labels indexed:', len(index))
    noiface = [k for k, v in index.items() if not v['elements']]
    print('labels with no extracted interface:', len(noiface))
    print('  e.g.', noiface[:15])


if __name__ == '__main__':
    main()
