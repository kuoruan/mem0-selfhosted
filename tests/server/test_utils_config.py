"""Tests for server/utils/config.py — expand_env_vars, is_truthy, load_json_config, merge_config."""

import json
import logging
from pathlib import Path

import pytest

# Import from the server package via its full module path.
from server.utils.config import expand_env_vars, is_truthy, load_json_config, merge_config


# ============================================================================
# is_truthy
# ============================================================================


class TestIsTruthy:
    """Boolean-string coercion shared across the server."""

    # --- bools pass through ---
    @pytest.mark.parametrize("value", [True])
    def test_bool_true(self, value: bool) -> None:
        assert is_truthy(value) is True

    def test_bool_false(self) -> None:
        assert is_truthy(False) is False

    # --- affirmative strings ---
    @pytest.mark.parametrize("value", ["1", "true", "True", "TRUE", "yes", "Yes", "on", "ON"])
    def test_affirmative_strings(self, value: str) -> None:
        assert is_truthy(value) is True

    # --- negative strings ---
    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "random", "enabled"])
    def test_non_affirmative_strings(self, value: str) -> None:
        assert is_truthy(value) is False

    # --- None / other types ---
    def test_none_returns_false(self) -> None:
        assert is_truthy(None) is False

    @pytest.mark.parametrize("value", [1, 1.0, -1, 0.5])
    def test_truthy_numbers(self, value: int | float) -> None:
        assert is_truthy(value) is True

    @pytest.mark.parametrize("value", [0, 0.0])
    def test_falsy_numbers(self, value: int | float) -> None:
        assert is_truthy(value) is False


# ============================================================================
# expand_env_vars
# ============================================================================


class TestExpandEnvVars:
    """Recursive environment-variable expansion."""

    def test_string_with_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_HOST", "db.example.com")
        assert expand_env_vars("$MY_HOST") == "db.example.com"

    def test_string_with_braces(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PORT", "5432")
        assert expand_env_vars("${PORT}") == "5432"

    def test_nested_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOST", "localhost")
        assert expand_env_vars({"db": {"host": "$HOST", "port": 5432}}) == {
            "db": {"host": "localhost", "port": 5432},
        }

    def test_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ITEM", "expanded")
        assert expand_env_vars(["$ITEM", "plain", 42]) == ["expanded", "plain", 42]

    def test_scalar_passthrough(self) -> None:
        assert expand_env_vars(123) == 123
        assert expand_env_vars(3.14) == 3.14
        assert expand_env_vars(True) is True
        assert expand_env_vars(None) is None

    def test_undefined_env_unchanged(self) -> None:
        # Undefined var stays as-is (os.path.expandvars behavior)
        assert expand_env_vars("$UNDEFINED_VAR_XYZ") == "$UNDEFINED_VAR_XYZ"


# ============================================================================
# load_json_config
# ============================================================================


class TestLoadJsonConfig:
    """Loading JSON config with env-var expansion."""

    def test_valid_file(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.json"
        cfg.write_text('{"key": "value", "num": 42}')
        result = load_json_config(cfg)
        assert result == {"key": "value", "num": 42}

    def test_env_expansion_in_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SECRET_VAL", "hunter2")
        cfg = tmp_path / "cfg.json"
        cfg.write_text('{"password": "$SECRET_VAL"}')
        assert load_json_config(cfg) == {"password": "hunter2"}

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="not found"):
            load_json_config(tmp_path / "nope.json")

    def test_missing_file_silent(self, tmp_path: Path) -> None:
        assert load_json_config(tmp_path / "nope.json", silent=True) is None

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json}")
        with pytest.raises(RuntimeError, match="Failed to load"):
            load_json_config(bad)

    def test_invalid_json_silent(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{bad")
        assert load_json_config(bad, silent=True) is None

    def test_non_dict_root_raises(self, tmp_path: Path) -> None:
        arr = tmp_path / "arr.json"
        arr.write_text("[1, 2, 3]")
        with pytest.raises(RuntimeError, match="JSON object"):
            load_json_config(arr)

    def test_non_dict_root_silent(self, tmp_path: Path) -> None:
        arr = tmp_path / "arr.json"
        arr.write_text("[1, 2, 3]")
        assert load_json_config(arr, silent=True) is None

    def test_string_path(self, tmp_path: Path) -> None:
        cfg = tmp_path / "cfg.json"
        cfg.write_text('{"a": 1}')
        assert load_json_config(str(cfg)) == {"a": 1}


# ============================================================================
# merge_config
# ============================================================================


class TestMergeConfig:
    """Deep-merge two dicts."""

    def test_flat_override(self) -> None:
        base = {"a": 1, "b": 2}
        updates = {"b": 99, "c": 3}
        assert merge_config(base, updates) == {"a": 1, "b": 99, "c": 3}

    def test_deep_merge(self) -> None:
        base = {"db": {"host": "localhost", "port": 5432}}
        updates = {"db": {"port": 3306, "user": "admin"}}
        assert merge_config(base, updates) == {
            "db": {"host": "localhost", "port": 3306, "user": "admin"},
        }

    def test_does_not_mutate_base(self) -> None:
        base = {"x": {"y": 1}}
        merge_config(base, {"x": {"z": 2}})
        assert base == {"x": {"y": 1}}

    def test_updates_dict_over_scalar(self) -> None:
        base = {"x": "scalar"}
        updates = {"x": {"nested": True}}
        assert merge_config(base, updates) == {"x": {"nested": True}}

    def test_updates_scalar_over_dict(self) -> None:
        base = {"x": {"nested": True}}
        updates = {"x": "scalar"}
        assert merge_config(base, updates) == {"x": "scalar"}

    def test_empty_updates(self) -> None:
        base = {"a": 1}
        assert merge_config(base, {}) == {"a": 1}

    def test_empty_base(self) -> None:
        assert merge_config({}, {"a": 1}) == {"a": 1}


# ============================================================================
# expand_env_vars — raw_keys skip expansion for sensitive fields (F5)
# ============================================================================


class TestExpandEnvVarsRawKeys:
    """Sensitive leaf keys (e.g. client_secret) must not be $-expanded."""

    def test_raw_key_preserved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XYZ", "LEAKED")
        out = expand_env_vars({"client_secret": "abc$XYZdef", "note": "$XYZ"}, raw_keys={"client_secret"})
        assert out["client_secret"] == "abc$XYZdef"
        assert out["note"] == "LEAKED"

    def test_raw_key_nested(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XYZ", "LEAKED")
        data = {"oidc": {"providers": [{"client_secret": "$XYZ", "name": "n"}]}}
        out = expand_env_vars(data, raw_keys={"client_secret"})
        assert out["oidc"]["providers"][0]["client_secret"] == "$XYZ"
        assert out["oidc"]["providers"][0]["name"] == "n"

    def test_raw_key_in_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XYZ", "LEAKED")
        out = expand_env_vars(
            [{"client_secret": "$XYZ"}, {"other": "$XYZ"}], raw_keys={"client_secret"}
        )
        assert out[0]["client_secret"] == "$XYZ"
        assert out[1]["other"] == "LEAKED"

    def test_no_raw_keys_expands_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without raw_keys, behaviour is unchanged (backward compat)."""
        monkeypatch.setenv("XYZ", "LEAKED")
        assert expand_env_vars({"client_secret": "$XYZ"}) == {"client_secret": "LEAKED"}


# ============================================================================
# load_json_config — raw_keys (F5)
# ============================================================================


class TestLoadJsonConfigRawKeys:
    """load_json_config(raw_keys=...) preserves sensitive values verbatim."""

    def test_sensitive_field_not_expanded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XYZ", "LEAKED")
        cfg = tmp_path / "cfg.json"
        cfg.write_text(json.dumps({"client_secret": "abc$XYZdef", "note": "$XYZ"}))
        result = load_json_config(cfg, raw_keys={"client_secret"})
        assert result["client_secret"] == "abc$XYZdef"
        assert result["note"] == "LEAKED"

    def test_multiple_raw_keys(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XYZ", "LEAKED")
        cfg = tmp_path / "cfg.json"
        cfg.write_text(
            json.dumps({"jwt_secret": "$XYZ", "admin_api_key": "$XYZ", "note": "$XYZ"})
        )
        result = load_json_config(cfg, raw_keys={"jwt_secret", "admin_api_key"})
        assert result["jwt_secret"] == "$XYZ"
        assert result["admin_api_key"] == "$XYZ"
        assert result["note"] == "LEAKED"


# ============================================================================
# is_truthy — unrecognized value warning (F9)
# ============================================================================


class TestIsTruthyWarning:
    """Unrecognized non-empty values warn once to avoid silent misconfiguration."""

    def test_unrecognized_value_warns_and_returns_false(self, caplog: pytest.LogCaptureFixture) -> None:
        from server.utils.config import _IS_TRUTHY_WARNED

        _IS_TRUTHY_WARNED.discard("enabled")
        with caplog.at_level(logging.WARNING):
            result = is_truthy("enabled")
        assert result is False
        assert any("Unrecognized boolean value" in rec.message for rec in caplog.records)

    def test_recognized_values_do_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            assert is_truthy("true") is True
            assert is_truthy("false") is False
        assert not [r for r in caplog.records if "Unrecognized boolean value" in r.message]
