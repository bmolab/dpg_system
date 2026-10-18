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
        self.registrations = []  # (label, factory_class, data, line, enclosing def)
        self.called = set()      # every function name this file calls
        # the def each register_node call sits in (innermost, as ast.walk
        # reaches nested defs after their parents)
        encl = {}
        for fd in ast.walk(self.tree):
            if isinstance(fd, ast.FunctionDef):
                for c in ast.walk(fd):
                    encl[id(c)] = fd.name
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Call):
                f = n.func
                self.called.add(f.id if isinstance(f, ast.Name) else getattr(f, 'attr', None))
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
                self.registrations.append((label, cls, data, n.lineno, encl.get(id(n))))



_MISSING = object()


def decide(test, name):
    """True / False when an if-test depends only on the node's label
    (self.label == 'np.ones', label in [...], and/or/not of those), None when
    it cannot be decided statically."""
    if name is None:
        return None
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        d = decide(test.operand, name)
        return None if d is None else not d
    if isinstance(test, ast.BoolOp):
        ds = [decide(v, name) for v in test.values]
        if isinstance(test.op, ast.And):
            return False if False in ds else (True if all(d is True for d in ds) else None)
        return True if True in ds else (False if all(d is False for d in ds) else None)
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1):
        return None
    left = test.left
    is_label = (isinstance(left, ast.Name) and left.id == 'label') or \
               (isinstance(left, ast.Attribute) and left.attr == 'label'
                and isinstance(left.value, ast.Name) and left.value.id == 'self')
    if not is_label:
        return None
    v = lit(test.comparators[0])
    op = test.ops[0]
    if isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(v, str):
        return (name == v) == isinstance(op, ast.Eq)
    if isinstance(op, (ast.In, ast.NotIn)) and isinstance(v, (list, tuple, set)):
        return (name in v) == isinstance(op, ast.In)
    return None


def pruned(cd, name, skip=()):
    """Every ast node of a class in source order, as (node, env, conditional):
    env holds the string literals local names have at that point, and
    conditional is True inside a branch or loop that cannot be decided here.
    Branches that test the label are followed only the way the registered
    NAME goes, so a node that adds 'min'/'max' only for np.rand, or names its
    outlet after its label, is reported per name rather than as the union of
    all its branches. Methods named in skip (overridden by the subclass being
    collected, without calling up) are left out."""
    out = []

    def visit(n, env, cond):
        if isinstance(n, ast.FunctionDef):
            env = {}                      # each method starts fresh
        if isinstance(n, ast.If):
            d = decide(n.test, name)
            if d is True:
                for s in n.body:
                    visit(s, env, cond)
                return
            if d is False:
                for s in n.orelse:
                    visit(s, env, cond)
                return
            visit(n.test, env, cond)
            a, b = dict(env), dict(env)
            for s in n.body:
                visit(s, a, True)
            for s in n.orelse:
                visit(s, b, True)
            # a name both branches leave alike keeps its value; else unknown
            env.clear()
            env.update({k: v for k, v in a.items() if b.get(k, _MISSING) == v})
            return
        out.append((n, dict(env), cond))
        loop = isinstance(n, (ast.For, ast.While, ast.ExceptHandler, ast.IfExp))
        for c in ast.iter_child_nodes(n):
            if n is cd and isinstance(c, ast.FunctionDef) and c.name in skip:
                continue
            visit(c, env, cond or loop)
        if isinstance(n, ast.Assign) and len(n.targets) == 1 \
                and isinstance(n.targets[0], ast.Name):
            v = lit(n.value)
            if isinstance(v, str):
                env[n.targets[0].id] = v
            else:
                env.pop(n.targets[0].id, None)

    visit(cd, {}, False)
    return out


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


def overrides(cd):
    """Methods a class defines that do not call the version they replace
    (super().m(...) or Base.m(self, ...)) -- a base's copy never runs."""
    own = set()
    for fd in cd.body:
        if not isinstance(fd, ast.FunctionDef):
            continue
        calls_up = any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                       and c.func.attr == fd.name
                       and (isinstance(c.func.value, ast.Name)
                            or (isinstance(c.func.value, ast.Call)
                                and getattr(c.func.value.func, 'id', '') == 'super'))
                       for c in ast.walk(fd))
        if not calls_up:
            own.add(fd.name)
    return own


def collect(cls_name, files, seen=None, depth=0, prefer=None, name=None, skip=frozenset()):
    """Walk a class and its bases, gathering declared interface elements.
    name is the registered label, used to follow label-dependent branches;
    skip is the methods a subclass below has already replaced."""
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
            out += collect(b, files, seen, depth + 1, prefer=fi.path, name=name,
                           skip=skip | overrides(cd))
        nodes = pruned(cd, name, skip)
        for n, env, cond in nodes:
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)):
                continue
            kind = ADDERS.get(n.func.attr)
            if kind is None:
                continue
            label = lit(n.args[0]) if n.args else None
            if label is None and n.args and isinstance(n.args[0], ast.Name):
                label = env.get(n.args[0].id)     # out_label = 'array of ones'
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
                'conditional': cond,
            })
        # help_file_name is often set on a BASE class (TorchDistributionNode
        # carries 't.dist_help' for every t.dist.* node), so record it here and
        # let collect()'s base walk carry it down to the subclasses.
        for n, _, _ in nodes:
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
        for n, _, _ in nodes:
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
        for n, _, _ in nodes:
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
        for n, _, _ in nodes:
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
    # A registration only counts if the app runs it: dpg_app calls
    # register_<module> for each module it imports, and other register_*
    # functions are called from those. A copy left in a module nobody
    # registers (lighting_nodes' color_source) must not overwrite the live one.
    app_src = open(os.path.join(ROOT, 'dpg_app.py')).read()
    called = set().union(*(fi.called for fi in files.values()))

    def live(fi, fn):
        stem = os.path.splitext(os.path.basename(fi.path))[0]
        return fn is None or fn in called or \
            (fn == 'register_' + stem and f"'{stem}'" in app_src)

    regs = [(fi, r) for fi in files.values() for r in fi.registrations]
    regs.sort(key=lambda x: live(x[0], x[1][4]))      # live ones written last
    where = collections.defaultdict(list)    # label -> every place registering it
    for fi, (label, cls, data, line, fn) in regs:
        where[label].append(f'{os.path.relpath(fi.path, ROOT)}:{line} ({cls}'
                            f'{"" if live(fi, fn) else ", never registered: " + str(fn) + " is not called"})')
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
            'elements': collect(cls, files, prefer=fi.path, name=label),
        }
    json.dump(index, open(sys.argv[1] if len(sys.argv) > 1 else 'iface.json', 'w'), indent=1)
    print('labels indexed:', len(index))
    noiface = [k for k, v in index.items() if not v['elements']]
    print('labels with no extracted interface:', len(noiface))
    print('  e.g.', noiface[:15])
    # A name registered twice: the app keeps both factories, the node list
    # shows it twice, and the first one registered builds the node -- an
    # accident of import order.
    dups = {k: v for k, v in where.items() if len(v) > 1}
    print('labels registered more than once:', len(dups))
    for k, v in sorted(dups.items()):
        print(f'  {k}:', '; '.join(v))


if __name__ == '__main__':
    main()
