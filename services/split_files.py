"""Split Model/ and App/ into separate per-microservice files, plus standalone service entrypoints."""

import ast
from collections import defaultdict
from pathlib import Path

from services.catalog import ALL_SERVICES, owner_of

ROOT = Path(__file__).resolve().parents[1]


def _parse_routes(src: str, prefix: str, obj_name: str):
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    items = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            decs = [d.lineno for d in node.decorator_list]
            start = (min(decs) if decs else node.lineno) - 1
            end = node.end_lineno
            routes = [
                prefix + d.args[0].value
                for d in node.decorator_list
                if isinstance(d, ast.Call)
                and isinstance(d.func, ast.Attribute)
                and d.func.attr == "route"
            ]
            calls = [
                c.func.attr
                for c in ast.walk(node)
                if isinstance(c, ast.Call)
                and isinstance(c.func, ast.Attribute)
                and isinstance(c.func.value, ast.Name)
                and c.func.value.id == obj_name
            ]
            svc = owner_of(routes[0])
            block = "".join(lines[start:end])
            items.append((svc, routes[0], node.name, calls, block))
    return items


def _extract_node(lines, node):
    decs = [d.lineno for d in getattr(node, "decorator_list", [])]
    start = (min(decs) if decs else node.lineno) - 1
    return "".join(lines[start : node.end_lineno])


def _dedup_attrs(attrs):
    seen = {}
    for a in attrs:
        if isinstance(a, ast.Assign) and len(a.targets) == 1 and isinstance(a.targets[0], ast.Name):
            seen[a.targets[0].id] = a
        else:
            seen[id(a)] = a
    return sorted(seen.values(), key=lambda x: x.lineno)


def main():
    reg_route_src = (ROOT / "App" / "netwirk_registrationRoute.py").read_text(encoding="utf-8")
    prof_route_src = (ROOT / "App" / "netwirkProfileRoute.py").read_text(encoding="utf-8")

    reg_routes = _parse_routes(reg_route_src, "/registrationRoute", "register")
    prof_routes = _parse_routes(prof_route_src, "/netwirkProfileRoute", "netwirkProfile")

    # 1. Parse Model/netwirk_registration.py
    m_reg_src = (ROOT / "Model" / "netwirk_registration.py").read_text(encoding="utf-8")
    m_reg_lines = m_reg_src.splitlines(keepends=True)
    m_reg_tree = ast.parse(m_reg_src)
    reg_cls = [n for n in m_reg_tree.body if isinstance(n, ast.ClassDef)][0]
    reg_header = "".join(m_reg_lines[: reg_cls.lineno - 1])
    reg_methods = {n.name: n for n in reg_cls.body if isinstance(n, ast.FunctionDef)}
    reg_attrs = [n for n in reg_cls.body if not isinstance(n, ast.FunctionDef)]

    reg_calls = {
        m: {
            a.attr
            for a in ast.walk(node)
            if isinstance(a, ast.Attribute)
            and isinstance(a.value, ast.Name)
            and a.value.id == "self"
            and a.attr in reg_methods
        }
        - {m}
        for m, node in reg_methods.items()
    }
    reg_used = defaultdict(set)
    for svc, _, _, calls, _ in reg_routes:
        for c in calls:
            stack = [c]
            seen = set()
            while stack:
                cur = stack.pop()
                if cur in seen or cur not in reg_methods:
                    continue
                seen.add(cur)
                reg_used[cur].add(svc)
                stack.extend(reg_calls.get(cur, ()))

    # 2. Parse Model/netwirk_profile.py
    m_prof_src = (ROOT / "Model" / "netwirk_profile.py").read_text(encoding="utf-8")
    m_prof_lines = m_prof_src.splitlines(keepends=True)
    m_prof_tree = ast.parse(m_prof_src)
    prof_cls = [n for n in m_prof_tree.body if isinstance(n, ast.ClassDef)][0]
    prof_header = "".join(m_prof_lines[: prof_cls.lineno - 1])
    prof_methods = {n.name: n for n in prof_cls.body if isinstance(n, ast.FunctionDef)}
    prof_attrs = [n for n in prof_cls.body if not isinstance(n, ast.FunctionDef)]

    prof_calls = {
        m: {
            a.attr
            for a in ast.walk(node)
            if isinstance(a, ast.Attribute)
            and isinstance(a.value, ast.Name)
            and a.value.id == "self"
            and a.attr in prof_methods
        }
        - {m}
        for m, node in prof_methods.items()
    }
    prof_used = defaultdict(set)
    for svc, _, _, calls, _ in prof_routes:
        for c in calls:
            stack = [c]
            seen = set()
            while stack:
                cur = stack.pop()
                if cur in seen or cur not in prof_methods:
                    continue
                seen.add(cur)
                prof_used[cur].add(svc)
                stack.extend(prof_calls.get(cur, ()))

    for c in [
        "_batch_fetch_coords",
        "_batch_get_user_tag_counts",
        "_batch_get_places",
        "_cosine_similarity_fast",
        "_age_from_dob",
        "_load_embedding",
        "_haversine_km",
        "_load_vibr_config",
    ]:
        stack = [c]
        seen = set()
        while stack:
            cur = stack.pop()
            if cur in seen or cur not in prof_methods:
                continue
            seen.add(cur)
            prof_used[cur].add("recsys")
            stack.extend(prof_calls.get(cur, ()))

    for c in ["_resolve_vibr_config_path", "_load_vibr_config", "_get_openai_api_key"]:
        prof_used[c].update(["profile", "geo", "social", "recsys", "legacy"])

    reg_by_bucket = defaultdict(list)
    for m, node in sorted(reg_methods.items(), key=lambda kv: kv[1].lineno):
        if m == "__init__" or len(reg_used.get(m, ())) > 1:
            reg_by_bucket["base"].append(node)
        elif len(reg_used.get(m, ())) == 1:
            reg_by_bucket[next(iter(reg_used[m]))].append(node)
        else:
            reg_by_bucket["legacy"].append(node)

    prof_by_bucket = defaultdict(list)
    for m, node in sorted(prof_methods.items(), key=lambda kv: kv[1].lineno):
        if m == "__init__" or len(prof_used.get(m, ())) > 1:
            prof_by_bucket["base"].append(node)
        elif len(prof_used.get(m, ())) == 1:
            prof_by_bucket[next(iter(prof_used[m]))].append(node)
        elif node.lineno >= 15800:
            prof_by_bucket["recsys"].append(node)
        else:
            prof_by_bucket["legacy"].append(node)

    common_code = (
        '"""Shared base classes, constants, and cross-service helpers."""\n\n'
        + reg_header
        + "\n"
        + prof_header
        + "\n\n"
        + "class RegistrationBase:\n"
        + "".join(_extract_node(m_reg_lines, a) for a in _dedup_attrs(reg_attrs))
        + "\n"
        + "\n".join(_extract_node(m_reg_lines, n) for n in reg_by_bucket["base"])
        + "\n\n\n"
        + "class ProfileBase:\n"
        + "".join(_extract_node(m_prof_lines, a) for a in _dedup_attrs(prof_attrs))
        + "\n"
        + "\n".join(_extract_node(m_prof_lines, n) for n in prof_by_bucket["base"])
        + "\n\n"
        + "# Alias used by static methods referencing profile.<attr>\n"
        + "profile = ProfileBase\n"
    )
    (ROOT / "Model" / "common_base.py").write_text(common_code, encoding="utf-8", newline="\n")

    auth_model_code = (
        '"""1. Auth & Identity microservice model."""\n\n'
        + reg_header
        + "\n"
        + prof_header
        + "\n"
        + "from Model.common_base import ProfileBase, RegistrationBase, profile\n\n\n"
        + "class AuthRegistrationService(RegistrationBase):\n"
        + "\n".join(_extract_node(m_reg_lines, n) for n in reg_by_bucket["auth"])
        + "\n\n\n"
        + "class AuthProfileService(ProfileBase):\n"
        + "\n".join(_extract_node(m_prof_lines, n) for n in prof_by_bucket["auth"])
        + "\n"
    )
    (ROOT / "Model" / "auth_service.py").write_text(auth_model_code, encoding="utf-8", newline="\n")

    for svc_name, cls_name, title in [
        ("profile", "ProfileService", "2. Profile & Persona microservice model."),
        ("geo", "GeoService", "3. Geo & Check-in microservice model."),
        ("social", "SocialService", "4. Social Graph microservice model."),
        ("recsys", "RecsysService", "6. RecSys & AI microservice model."),
    ]:
        code = (
            f'"""{title}"""\n\n'
            + prof_header
            + "\n"
            + "from Model.common_base import ProfileBase, profile\n\n\n"
            + f"class {cls_name}(ProfileBase):\n"
            + "\n".join(_extract_node(m_prof_lines, n) for n in prof_by_bucket[svc_name])
            + "\n"
        )
        (ROOT / "Model" / f"{svc_name}_service.py").write_text(code, encoding="utf-8", newline="\n")

    legacy_model_code = (
        '"""Legacy microservice model for remaining endpoints."""\n\n'
        + reg_header
        + "\n"
        + prof_header
        + "\n"
        + "from Model.common_base import ProfileBase, RegistrationBase, profile\n\n\n"
        + "class LegacyRegistrationService(RegistrationBase):\n"
        + "\n".join(_extract_node(m_reg_lines, n) for n in reg_by_bucket["legacy"])
        + "\n\n\n"
        + "class LegacyProfileService(ProfileBase):\n"
        + "\n".join(_extract_node(m_prof_lines, n) for n in prof_by_bucket["legacy"])
        + "\n"
    )
    (ROOT / "Model" / "legacy_service.py").write_text(
        legacy_model_code, encoding="utf-8", newline="\n"
    )

    auth_reg_blocks = [
        b.replace("@registrationRoute.route(", "@authRegistrationRoute.route(")
        for svc, _, _, _, b in reg_routes
        if svc == "auth"
    ]
    auth_prof_blocks = [
        b.replace("@netwirkProfileRoute.route(", "@authProfileRoute.route(")
        for svc, _, _, _, b in prof_routes
        if svc == "auth"
    ]
    auth_routes_code = (
        '"""1. Auth & Identity microservice routes."""\n'
        "import logging\n"
        "from flask import Blueprint, jsonify, request\n"
        "from flasgger import swag_from\n"
        "from Model.auth_service import AuthProfileService as profile, AuthRegistrationService as registration\n\n"
        "authRegistrationRoute = Blueprint('registrationRoute', __name__, url_prefix='/registrationRoute')\n"
        "authProfileRoute = Blueprint('authProfileRoute', __name__, url_prefix='/netwirkProfileRoute')\n"
        "BLUEPRINTS = (authRegistrationRoute, authProfileRoute)\n\n"
        + "\n".join(auth_reg_blocks)
        + "\n"
        + "\n".join(auth_prof_blocks)
    )
    (ROOT / "App" / "auth_routes.py").write_text(auth_routes_code, encoding="utf-8", newline="\n")

    for svc_name, cls_name, bp_name, title in [
        ("profile", "ProfileService", "profileServiceRoute", "2. Profile & Persona microservice routes."),
        ("geo", "GeoService", "geoServiceRoute", "3. Geo & Check-in microservice routes."),
        ("social", "SocialService", "socialServiceRoute", "4. Social Graph microservice routes."),
        ("recsys", "RecsysService", "recsysServiceRoute", "6. RecSys & AI microservice routes."),
    ]:
        blocks = [
            b.replace("@netwirkProfileRoute.route(", f"@{bp_name}.route(")
            for svc, _, _, _, b in prof_routes
            if svc == svc_name
        ]
        code = (
            f'"""{title}"""\n'
            "import logging\n"
            "from flask import Blueprint, jsonify, request\n"
            "from flasgger import swag_from\n"
            f"from Model.{svc_name}_service import {cls_name} as profile\n\n"
            f"{bp_name} = Blueprint('{bp_name}', __name__, url_prefix='/netwirkProfileRoute')\n"
            f"BLUEPRINTS = ({bp_name},)\n\n"
            + "\n".join(blocks)
        )
        (ROOT / "App" / f"{svc_name}_routes.py").write_text(code, encoding="utf-8", newline="\n")

    leg_reg_blocks = [
        b.replace("@registrationRoute.route(", "@legacyRegistrationRoute.route(")
        for svc, _, _, _, b in reg_routes
        if svc == "legacy"
    ]
    leg_prof_blocks = [
        b.replace("@netwirkProfileRoute.route(", "@legacyProfileRoute.route(")
        for svc, _, _, _, b in prof_routes
        if svc == "legacy"
    ]
    legacy_routes_code = (
        '"""Legacy microservice routes for remaining endpoints."""\n'
        "import logging\n"
        "from flask import Blueprint, jsonify, request\n"
        "from flasgger import swag_from\n"
        "from App.matchDemoRoute import matchDemoRoute\n"
        "from Model.legacy_service import LegacyProfileService as profile, LegacyRegistrationService as registration\n\n"
        "legacyRegistrationRoute = Blueprint('legacyRegistrationRoute', __name__, url_prefix='/registrationRoute')\n"
        "legacyProfileRoute = Blueprint('legacyProfileRoute', __name__, url_prefix='/netwirkProfileRoute')\n"
        "BLUEPRINTS = (legacyRegistrationRoute, legacyProfileRoute, matchDemoRoute)\n\n"
        + "\n".join(leg_reg_blocks)
        + "\n"
        + "\n".join(leg_prof_blocks)
    )
    (ROOT / "App" / "legacy_routes.py").write_text(legacy_routes_code, encoding="utf-8", newline="\n")

    # Standalone runnable app files per microservice in services/<name>_app.py
    for svc in ALL_SERVICES:
        svc_app_code = (
            f'"""Standalone Flask entrypoint for the {svc!r} microservice."""\n'
            "import os\n\n"
            f'os.environ["SERVICE_NAME"] = "{svc}"\n\n'
            "from app import app  # noqa: E402,F401\n\n"
            'if __name__ == "__main__":\n'
            '    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))\n'
        )
        (ROOT / "services" / f"{svc}_app.py").write_text(
            svc_app_code, encoding="utf-8", newline="\n"
        )

    # Backwards-compatible Model/netwirk_registration.py and Model/netwirk_profile.py
    (ROOT / "Model" / "netwirk_registration.py").write_text(
        '"""Combined registration model (imports Auth & Legacy registration services)."""\n\n'
        "from Model.auth_service import AuthRegistrationService\n"
        "from Model.legacy_service import LegacyRegistrationService\n\n\n"
        "class registration(AuthRegistrationService, LegacyRegistrationService):\n"
        "    pass\n",
        encoding="utf-8",
        newline="\n",
    )
    (ROOT / "Model" / "netwirk_profile.py").write_text(
        '"""Combined profile model (imports all split microservice profile classes)."""\n\n'
        "from Model.auth_service import AuthProfileService\n"
        "from Model.common_base import (\n"
        "    ALL_TAGS,\n"
        "    CHECK_INTERVAL,\n"
        "    COMM_COMPAT,\n"
        "    COMM_STYLE_MAP,\n"
        "    DEFAULT_WEIGHTS,\n"
        "    INACTIVITY_TIMEOUT_SECONDS,\n"
        "    INTENT_COMPAT,\n"
        "    INTENT_MAP,\n"
        "    IST,\n"
        "    MATCHING_WEIGHTS_V2,\n"
        "    MIN_INTERACTIONS_FOR_DYNAMIC_WEIGHTS,\n"
        "    RECENCY_HALF_LIFE_DAYS,\n"
        "    SKIP_PENALTY_WEIGHT,\n"
        "    SUPPORTED_METROS,\n"
        "    TAG_TYPE_MAP,\n"
        ")\n"
        "from Model.geo_service import GeoService\n"
        "from Model.legacy_service import LegacyProfileService\n"
        "from Model.profile_service import ProfileService\n"
        "from Model.recsys_service import RecsysService\n"
        "from Model.social_service import SocialService\n\n\n"
        "class profile(\n"
        "    AuthProfileService,\n"
        "    ProfileService,\n"
        "    GeoService,\n"
        "    SocialService,\n"
        "    RecsysService,\n"
        "    LegacyProfileService,\n"
        "):\n"
        "    pass\n",
        encoding="utf-8",
        newline="\n",
    )

    print("Successfully generated per-microservice Model/, App/, and services/*_app.py files.")


if __name__ == "__main__":
    main()
