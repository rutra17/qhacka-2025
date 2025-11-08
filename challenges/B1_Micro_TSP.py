import nbclient, nbformat, types, pathlib, ast
NB_PATH = pathlib.Path(__file__).with_name("B1_Micro_TSP.ipynb")

def _extract_defs_imports(src: str):
    try:
        tree = ast.parse(src)
    except Exception:
        return None
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            keep.append(node)
    mod = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(mod)
    return compile(mod, f"{NB_PATH.name}::<filtered>", "exec")

def _load():
    nb = nbformat.read(NB_PATH, as_version=4)
    ns: dict = {}
    current_section = ""
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            lines = (cell.source or "").splitlines()
            if lines:
                first = lines[0].lstrip()
                if first.startswith("## "):
                    current_section = first[3:].strip().lower()
        elif cell.cell_type == "code":
            sec = (current_section or "").lower()
            code = (cell.source or "").strip()
            if not code:
                continue
            if "public tests" in sec:
                continue
            if "baseline" in sec:
                filtered = _extract_defs_imports(code)
                if filtered is not None:
                    exec(filtered, ns, ns)
                continue
            exec(code, ns, ns)
    return ns

_ns = _load()
solve = _ns.get("solve")
if solve is None:
    raise ImportError("`solve` not found in notebook: B1_Micro_TSP.ipynb")
