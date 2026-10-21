"""Sync route files with model files: remove any route whose model method was removed."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ROUTE_TO_MODEL = [
    ("App/legacy_routes.py", ["Model/legacy_service.py"]),
    ("App/auth_routes.py", ["Model/auth_service.py"]),
    ("App/profile_routes.py", ["Model/profile_service.py"]),
    ("App/geo_routes.py", ["Model/geo_service.py"]),
    ("App/social_routes.py", ["Model/social_service.py"]),
    ("App/recsys_routes.py", ["Model/recsys_service.py"]),
    (
        "App/netwirkProfileRoute.py",
        [
            "Model/auth_service.py",
            "Model/profile_service.py",
            "Model/geo_service.py",
            "Model/social_service.py",
            "Model/recsys_service.py",
            "Model/legacy_service.py",
        ],
    ),
    (
        "App/netwirk_registrationRoute.py",
        ["Model/auth_service.py", "Model/legacy_service.py"],
    ),
]


def _defined_methods(model_paths):
    methods = set()
    base_tree = ast.parse((ROOT / "Model" / "common_base.py").read_text(encoding="utf-8"))
    for cls in [n for n in base_tree.body if isinstance(n, ast.ClassDef)]:
        methods.update(n.name for n in cls.body if isinstance(n, ast.FunctionDef))

    for rel in model_paths:
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            methods.update(n.name for n in cls.body if isinstance(n, ast.FunctionDef))
    return methods


def sync_route_file(route_rel: str, model_paths: list[str]) -> list[str]:
    route_path = ROOT / route_rel
    src = route_path.read_text(encoding="utf-8")
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    available = _defined_methods(model_paths)

    remove_ranges = []
    removed_routes = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        calls = [
            c.func.attr
            for c in ast.walk(node)
            if isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and isinstance(c.func.value, ast.Name)
            and c.func.value.id in ("register", "netwirkProfile")
        ]
        if calls and any(c not in available for c in calls):
            decs = [d.lineno for d in node.decorator_list]
            start = (min(decs) if decs else node.lineno) - 1
            end = node.end_lineno
            while end < len(lines) and lines[end].strip() == "":
                end += 1
            remove_ranges.append((start, end))
            routes = [
                d.args[0].value
                for d in node.decorator_list
                if isinstance(d, ast.Call)
                and isinstance(d.func, ast.Attribute)
                and d.func.attr == "route"
            ]
            removed_routes.extend(routes or [node.name])

    if remove_ranges:
        keep = []
        idx = 0
        for start, end in sorted(remove_ranges):
            keep.extend(lines[idx:start])
            idx = end
        keep.extend(lines[idx:])
        route_path.write_text("".join(keep), encoding="utf-8", newline="\n")

    return removed_routes


def remove_unused_yml_files() -> list[str]:
    used_ymls = set()
    for rf in (ROOT / "App").glob("*.py"):
        tree = ast.parse(rf.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "swag_from"
                and node.args
            ):
                used_ymls.add("App/" + node.args[0].value)

    removed = []
    for yml_path in sorted((ROOT / "App").rglob("*.yml")):
        rel = yml_path.relative_to(ROOT).as_posix()
        if rel not in used_ymls:
            yml_path.unlink()
            removed.append(rel)
    return removed


def main():
    total = 0
    for route_rel, model_paths in ROUTE_TO_MODEL:
        removed = sync_route_file(route_rel, model_paths)
        if removed:
            total += len(removed)
            print(f"{route_rel}: removed {len(removed)} route(s): {', '.join(removed)}")
    removed_ymls = remove_unused_yml_files()
    if removed_ymls:
        print(f"Removed {len(removed_ymls)} unused Swagger .yml file(s): {', '.join(removed_ymls)}")
    if total == 0 and not removed_ymls:
        print("All route files and Swagger .yml files are in sync with their model files.")


if __name__ == "__main__":
    main()
