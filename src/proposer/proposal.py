"""Proposal cards: what a propose tool hands to the user.

A proposal is structured data that a client can show as a card and, if the
user accepts it, apply to the thing being described:

    id          unique per card
    kind        the pack's name for this sort of proposal ("accessions", "pin")
    title       one line
    note        what was checked and what was found, worded by the server
    source      where the model says the values came from
    links       URL checks shown on the card: {url, ok, status, size, error}
    more        a line standing for the links not shown, or None
    action      label of the apply button
    applicable  False when nothing in it can be applied (no link reachable)
    payload     the values, in the shape the pack's client applies

The model only supplies values. The server validates them, checks the URLs,
leaves out what failed, decides `applicable` and words the note. What goes
back to the model is a one-line summary of what the user was shown.
"""
import re
import uuid

from pydantic import BaseModel, Field, ValidationError, field_validator

from . import net

SAMPLE_LINKS = 6       # links shown on a card with many
MAX_FAILED_SHOWN = 4   # of those, failed ones first


def clip(text, limit):
    text = re.sub(r'\s+', ' ', text or '').strip()
    return text if len(text) <= limit else text[:limit - 1].rstrip() + '…'


def plural(count: int, word: str) -> str:
    return '%d %s%s' % (count, word, '' if count == 1 else 's')


def make(kind, title, note, source, links, more, action, applicable, payload) -> dict:
    return {'id': uuid.uuid4().hex, 'kind': kind, 'title': clip(title, 120),
            'note': clip(note, 240), 'source': clip(source, 300), 'links': links,
            'more': more, 'action': action, 'applicable': bool(applicable),
            'payload': payload}


def sample_links(checks: list[dict]) -> tuple[list[dict], str | None]:
    """Up to SAMPLE_LINKS checks (failed first) and the 'N more links' line."""
    if len(checks) <= SAMPLE_LINKS:
        return checks, None
    failed = [c for c in checks if not c['ok']]
    shown = failed[:MAX_FAILED_SHOWN]
    shown += [c for c in checks if c['ok']][:SAMPLE_LINKS - len(shown)]
    rest = [c for c in checks if c not in shown]
    bad = sum(1 for c in rest if not c['ok'])
    if not bad:
        more = '%s more, all reachable.' % plural(len(rest), 'link')
    elif bad == len(rest):
        more = '%s more, none reachable.' % plural(len(rest), 'link')
    else:
        more = '%s more: %d reachable, %d not reachable.' % (
            plural(len(rest), 'link'), len(rest) - bad, bad)
    return shown, more


async def checks(urls: list[str]) -> dict[str, dict]:
    """URL -> check, each URL checked once."""
    unique = list(dict.fromkeys(urls))
    return {c['url']: c for c in await net.check_urls(unique)} if unique else {}


class SourcedArgs(BaseModel):
    """Base for the arguments of a propose tool: the model must say where the
    values came from, so the user can check them."""
    model_config = {'extra': 'ignore'}
    source: str = Field(min_length=3, description='Where the values came from, e.g. '
                        '"Data availability, para. 2 · Table S1" or a folder URL. Required.')
    title: str | None = None
    note: str | None = None

    @field_validator('source')
    @classmethod
    def _source(cls, value):
        value = value.strip()
        if len(value) < 3:
            raise ValueError('say where the values came from, e.g. '
                             '"Data availability, para. 2 · Table S1"')
        return value


def validation_message(error: ValidationError) -> str:
    """Pydantic errors in a form a small model can act on."""
    parts = []
    for item in error.errors()[:6]:
        where = '.'.join(str(p) for p in item['loc']) or 'arguments'
        parts.append('%s: %s' % (where, item['msg']))
    more = len(error.errors()) - 6
    return '; '.join(parts) + (' (and %d more)' % more if more > 0 else '')
