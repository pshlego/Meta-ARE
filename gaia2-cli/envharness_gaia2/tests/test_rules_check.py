from __future__ import annotations

import json
from pathlib import Path

import pytest

from envharness.core.types import Action

from envharness_gaia2.rules_check import RulesCheckError, check_rules_code, setup_commands


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
        (GOOD.replace("import json", "import os"), "`os` not allowed"),
        (GOOD + "\n        open('x')\n", None),
        ("class _Rules(Rules):\n    def step(self, a):\n        return a\n", "overrides ['step']"),
        ("class Other(Rules):\n    pass\n", "_Rules"),
        ("class _Rules(Rules):\n    def filter_action(self, a, s):\n        return a.__class__\n", "dunder"),
        (GOOD.replace("contacts", "scenario_universe_29_abc"), "scenario identifier"),
        ("x" * 9000, "over 8192 bytes"),
        # json.loads without `import json`: the hook's own try/except would swallow the NameError
        (GOOD.replace("import json\n", "").replace("except ValueError", "except Exception"), "add `import json`"),
        ("class _Rules(Rules):\n    def filter_action(self, a, s):\n        return helper(a)\n", "`helper`"),
    ],
)
def test_rules_check_rejects(code, message):
    with pytest.raises(RulesCheckError) as exc:
        check_rules_code(code)
    if message:
        assert message in str(exc.value)


def test_undefined_name_check_accepts_locally_bound_names():
    check_rules_code(
        "import re\n\nclass _Rules(Rules):\n"
        "    def filter_action(self, action, env_state):\n"
        "        argv = [t for t in action.kwargs.get('argv', []) if (m := re.match(r'--', t)) or t]\n"
        "        try:\n            n = int(argv[0])\n        except Exception as exc:\n            n = len(str(exc))\n"
        "        return action if n or Blocked else Action(name=action.name, kwargs={**action.kwargs})\n"
    )


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


def test_transition_hook_is_allowed_and_runs_in_the_container_loader():
    code = (
        "class _Rules(Rules):\n"
        "    def modify_transition(self, action, response, env_state):\n"
        "        if response.info['wrote']:\n"
        "            return response\n"
        "        return EnvResponse(observation=Observation(text=''), info=dict(response.info, exit_code=1))\n"
    )
    check_rules_code(code)
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "env_rules_t", Path(__file__).resolve().parents[2] / "cli" / "gaia2_cli" / "env_rules.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["env_rules_t"] = module
    spec.loader.exec_module(module)
    rules = module.load_rules_class(code)()
    response = module.EnvResponse(observation=module.Observation("[]"), info={"exit_code": 0, "stderr": "", "wrote": False})
    assert rules.modify_transition(module.Action("contacts.get-contacts"), response, {}).info["exit_code"] == 1


def test_setup_commands_are_single_app_calls():
    ok = [Action(name="exec", kwargs={"command": "contacts add-new-contact --first-name 'A;B' --last-name C"})]
    assert setup_commands(ok) == ["contacts add-new-contact --first-name 'A;B' --last-name C"]
    for bad in ("ls /var/gaia2", "contacts get-contacts; cat /etc/passwd", "contacts get-contacts | sh",
                "calendar get-all-tags && rm x", "contacts get-contacts > /tmp/x"):
        with pytest.raises(RulesCheckError):
            setup_commands([Action(name="exec", kwargs={"command": bad})])
    with pytest.raises(RulesCheckError, match="exec"):
        setup_commands([Action(name="contacts", kwargs={"command": "contacts get-contacts"})])
    with pytest.raises(RulesCheckError, match="scenario"):
        setup_commands(ok, forbidden_ids=["Bxx", "A;B"])


def test_dunder_names_are_rejected_but_public_getattr_is_allowed():
    with pytest.raises(RulesCheckError, match="dunder name"):
        check_rules_code("class _Rules(Rules):\n    def filter_action(self, a, s):\n        return __builtins__['open']\n")
    check_rules_code("class _Rules(Rules):\n    def filter_action(self, a, s):\n        return getattr(s, 'app', a)\n")


def test_schema_states_every_limit_the_check_enforces():
    from envharness_gaia2.orchestrator import ENV_STATE_SCHEMA
    from envharness_gaia2.rules_check import ALLOWED_IMPORTS, MAX_BYTES, MAX_SETUP_ACTIONS

    assert f"over {MAX_BYTES} bytes" in ENV_STATE_SCHEMA
    assert f"at most {MAX_SETUP_ACTIONS}" in ENV_STATE_SCHEMA
    assert all(name in ENV_STATE_SCHEMA for name in ALLOWED_IMPORTS)


@pytest.mark.parametrize(
    "body",
    [
        # escapes found by the audit: private module attributes, modules re-exported by an
        # allowed module, frame/generator internals, and getattr with a computed name
        "import random\n        return random._os.system('id')",
        "import json\n        return json.codecs.open('/etc/passwd').read()",
        "import json as j\n        m = j\n        return m.codecs",
        "from json import codecs\n        return codecs",
        "import textwrap\n        return textwrap.re",
        "return (i for i in [0]).gi_frame.f_globals",
        "return (i for i in [0]).gi_frame.f_back",
        "return getattr(a, 'gi' + '_frame')",
        "return getattr(a, 'f_globals')",
        "g = getattr\n        return g(a, 'x')",
        "return a._private",
    ],
)
def test_rules_check_rejects_sandbox_escapes(body):
    code = f"class _Rules(Rules):\n    def filter_action(self, a, s):\n        {body}\n"
    with pytest.raises(RulesCheckError):
        check_rules_code(code)


def test_rules_check_allows_self_private_helpers_and_module_functions():
    check_rules_code(
        "import json\nimport re\nfrom json import loads\n\nclass _Rules(Rules):\n"
        "    def _rank(self, rows):\n        return sorted(rows, key=lambda r: re.sub('a', '', json.dumps(r)))\n"
        "    def filter_observation(self, obs, s):\n        return Observation(text=json.dumps(self._rank(loads(obs.text))))\n"
    )


def test_rules_check_messages_lead_with_the_cause():
    # The designer sees only `err=...[:60]` of "RulesCheckError: <message>" (harness_agent.py).
    cases = {
        "x" * 9000: "over 8192 bytes",
        "import os\nclass _Rules(Rules):\n    pass\n": "import `os` not allowed",
        "class _Rules(Rules):\n    def filter_observation(self, o, s):\n        return json.loads(o.text)\n":
            "`json` undefined: add `import json`",
    }
    for code, head in cases.items():
        with pytest.raises(RulesCheckError) as exc:
            check_rules_code(code)
        assert f"RulesCheckError: {exc.value}"[:60].startswith(f"RulesCheckError: {head}"[:60])
