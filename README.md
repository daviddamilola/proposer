# proposer

Assistants that read and propose but never act.

A model is given a description of something to build (a paper, a folder of
files, a few URLs) and fills in the declarative input of a pipeline that
already knows how to build it. It may only read and propose. The server
validates every value and checks every link before the user sees a proposal,
and the user decides whether to apply it.

The engine is the same in every domain. A **pack** makes it an assistant for
one thing: what is being described, how to look at it, and what can be
proposed. See `src/proposer/pack.py`.

Packs: `tersect` (genomics datasets for Tersect Browser).

    uv sync
    uv run pytest
