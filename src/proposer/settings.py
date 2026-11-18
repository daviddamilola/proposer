"""Settings, read from the environment each time they are needed.

Every setting is PROPOSER_<NAME>. An application that embeds proposer and
already has its own variable names can keep them:

    from proposer import settings
    settings.alias('TB_ASSISTANT_', ALLOW_PRIVATE_URLS='TB_BUILDER_ALLOW_PRIVATE_URLS')

After that TB_ASSISTANT_URL is read where PROPOSER_URL would be (the alias
wins if both are set), and messages that name a variable name the alias.
"""
import os

_prefixes: list[str] = []
_names: dict[str, str] = {}


def alias(prefix: str = '', **names: str) -> None:
    """Also read settings from `prefix`<NAME>, and each NAME=variable given."""
    if prefix and prefix not in _prefixes:
        _prefixes.append(prefix)
    _names.update(names)


def reset() -> None:
    """Forget the aliases (tests)."""
    _prefixes.clear()
    _names.clear()


def variables(name: str) -> list[str]:
    """The environment variables that hold setting `name`, first wins."""
    found = [_names[name]] if name in _names else []
    return found + [prefix + name for prefix in _prefixes] + ['PROPOSER_' + name]


def variable(name: str) -> str:
    """The variable to name to the person who sets it."""
    return variables(name)[0]


def get(name: str, default: str = '') -> str:
    for candidate in variables(name):
        value = os.environ.get(candidate, '').strip()
        if value:
            return value
    return default
