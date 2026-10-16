from __future__ import annotations

import json
from pathlib import Path

import pytest

from envharness_gaia2.rules_check import RulesCheckError, check_rules_code


GOOD = """
import json

class _Rules(Rules):
    def filter_action(self, action, env_state):
        if env_state["app"] == "contacts" and env_state["call_index_app"] == 1:
            return Blocked(reason="Error: contacts service is temporarily unavailable, please retry.")
        return action

    def filter_observation(self, obs, env_state):
        try:
            items = json.loads(obs.text)
        except ValueError:
            return obs
        return Observation(text=json.dumps(items[:3])) if isinstance(items, list) else obs
"""


@pytest.mark.parametrize(
    ("code", "message"),
    [
        (GOOD.replace("import json", "import os"), "may import only"),
        (GOOD + "\n        open('x')\n", None),
        ("class _Rules(Rules):\n    def modify_transition(self, a, r, s):\n        return r\n", "modify_transition"),
        ("class Other(Rules):\n    pass\n", "_Rules"),
        ("class _Rules(Rules):\n    def filter_action(self, a, s):\n        return a.__class__\n", "dunder"),
        (GOOD.replace("contacts", "scenario_universe_29_abc"), "scenario identifiers"),
        ("x" * 9000, "larger"),
    ],
)
def test_rules_check_rejects(code, message):
    with pytest.raises(RulesCheckError) as exc:
        check_rules_code(code)
    if message:
        assert message in str(exc.value)


def test_rules_check_accepts_contract_and_matches_container_loader():
    check_rules_code(GOOD)
    check_rules_code("")  # pass-through
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "env_rules", Path(__file__).resolve().parents[2] / "cli" / "gaia2_cli" / "env_rules.py"
    )
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules["env_rules"] = module
    spec.loader.exec_module(module)
    rules = module.load_rules_class(GOOD)()
    blocked = rules.filter_action(module.Action("contacts.get-contacts"), {"app": "contacts", "call_index_app": 1})
    assert isinstance(blocked, module.Blocked)
    shown = rules.filter_observation(module.Observation(json.dumps(list(range(10)))), {"app": "emails"})
    assert json.loads(shown.text) == [0, 1, 2]
