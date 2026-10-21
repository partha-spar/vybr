import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

from services import gateway  # noqa: E402
from services.catalog import ALL_SERVICES, LEGACY, SERVICES, is_common, owner_of  # noqa: E402


def _build_app(service):
    os.environ["SERVICE_NAME"] = service or ""
    try:
        if "app" in sys.modules:
            module = importlib.reload(sys.modules["app"])
        else:
            module = importlib.import_module("app")
        return module.app
    finally:
        os.environ["SERVICE_NAME"] = ""


def _paths(flask_app):
    return {rule.rule for rule in flask_app.url_map.iter_rules() if not is_common(rule.rule)}


@pytest.fixture(scope="module")
def monolith_paths():
    return _paths(_build_app(None))


def test_every_catalog_path_exists_in_the_app(monolith_paths):
    missing = [p for paths in SERVICES.values() for p in paths if p not in monolith_paths]
    assert missing == []


def test_no_path_is_owned_by_two_services():
    seen = {}
    for name, paths in SERVICES.items():
        for path in paths:
            assert path not in seen, f"{path} listed in both {seen[path]} and {name}"
            seen[path] = name


@pytest.mark.parametrize("service", ALL_SERVICES)
def test_service_serves_only_its_own_routes(service, monolith_paths):
    service_app = _build_app(service)
    expected = {p for p in monolith_paths if owner_of(p) == service}
    assert _paths(service_app) == expected

    response = service_app.test_client().get("/health")
    assert response.status_code == 200


def test_services_together_cover_the_whole_app(monolith_paths):
    covered = set()
    for service in ALL_SERVICES:
        covered |= _paths(_build_app(service))
    assert covered == monolith_paths


def test_route_from_another_service_returns_404():
    auth_app = _build_app("auth")
    response = auth_app.test_client().post("/netwirkProfileRoute/get_user_list_offline", json={})
    assert response.status_code == 404


def test_unknown_service_name_fails_startup():
    with pytest.raises(SystemExit):
        _build_app("not-a-service")


def test_gateway_config_is_up_to_date():
    assert gateway.CONF_PATH.read_text(encoding="utf-8") == gateway.render(), (
        "Run `python -m services.gateway` to regenerate services/gateway/nginx.conf"
    )


def test_legacy_keeps_unlisted_endpoints(monolith_paths):
    assert owner_of("/demo") == LEGACY
    assert "/demo" in _paths(_build_app(LEGACY))


@pytest.mark.parametrize("service", ALL_SERVICES)
def test_per_service_route_and_model_files_are_self_contained(service):
    route_mod = importlib.import_module(f"App.{service}_routes")
    model_mod = importlib.import_module(f"Model.{service}_service")
    app_mod = importlib.import_module(f"services.{service}_app")
    assert hasattr(route_mod, "BLUEPRINTS") and len(route_mod.BLUEPRINTS) >= 1
    assert model_mod.__file__ is not None
    assert app_mod.app is not None

    for attr_name in ("profile", "registration"):
        cls = getattr(route_mod, attr_name, None)
        if cls is None:
            continue
        assert any(
            base.__module__ == f"Model.{service}_service" for base in cls.__mro__
        )


def test_every_route_calls_an_existing_model_method():
    from services.sync_routes import ROUTE_TO_MODEL, _defined_methods
    import ast

    for route_rel, model_paths in ROUTE_TO_MODEL:
        available = _defined_methods(model_paths)
        tree = ast.parse((ROOT / route_rel).read_text(encoding="utf-8"))
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
            missing = [c for c in calls if c not in available]
            assert not missing, f"{route_rel}:{node.name} calls missing model method(s): {missing}"

