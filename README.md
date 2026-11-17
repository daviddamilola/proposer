# proposer

Assistants that read and propose but never act.

Many pipelines are driven by a declarative input: a manifest, a config file, a
form. Filling it in is the slow part. Someone reads a paper or a download
page, finds the files, checks that the links work, and writes it all down in
the pipeline's own terms.

`proposer` gives that job to a language model without giving it the pipeline.
The model can only **read** (papers, folder listings, tables, the current
config) and **propose** values. The server validates every proposal, checks
every URL, drops what fails and words the result itself. The user sees a card
and decides whether to apply it. Nothing is saved, run or deleted by the
model.

It works with small local models (it was developed against `qwen3:8b` and
`qwen3:14b` on Ollama), because the server, not the model, is what makes a
proposal trustworthy.

## Packs

The engine is the same in every domain. A **pack** makes it an assistant for
one thing.

| Pack | Describes | A proposal is applied by |
|---|---|---|
| `tersect` | A genomics dataset for [Tersect Browser](https://github.com/Tersect-Browser/Tersect-browser): reference genome, one VCF per accession, build options | The dataset wizard's form |
| `opentargets` | New inputs for the [Open Targets pipeline](https://github.com/opentargets/pipeline): `copy` tasks and version pins in `pis/config.yaml` | A maintainer reviewing a patch |

## Try it

You need [uv](https://docs.astral.sh/uv/) and an OpenAI-compatible model
endpoint that supports tool calling, for example a local
[Ollama](https://ollama.com):

    uv sync
    export PROPOSER_URL=http://localhost:11434/v1
    export PROPOSER_MODEL=qwen3:14b

    uv run proposer chat --pack opentargets

Then describe what you want added:

    > Reactome also publishes gene sets at
      https://reactome.org/download/current/ReactomePathways.gmt.zip. Add it as an input.

    ┌─ [1] Add 1 input to reactome
    │ 1 reachable
    │ Source: https://reactome.org/download/current/ReactomePathways.gmt.zip
    │ ✓ https://reactome.org/download/current/ReactomePathways.gmt.zip 293.1K
    │
    │ +    - name: copy Reactome gene sets
    │ +      source: https://reactome.org/download/current/ReactomePathways.gmt.zip
    │ +      destination: input/reactome/ReactomePathways.gmt.zip
    └─
    Add to patch — "Add 1 input to reactome"? [y/N]

What you accept is collected into `out/opentargets-inputs.patch`, which
applies in the pipeline repository with `git apply`. By default the pack reads
`pis/config.yaml` from the `main` branch on GitHub; point it at a checkout
with `--set config=/path/to/pipeline/pis/config.yaml`.

To serve a pack over HTTP for a web client (server-sent events):

    uv run proposer serve --pack tersect

## Writing a pack

A pack is a subclass of `proposer.pack.Pack`:

| It says | With |
|---|---|
| What is being described, in a few lines | `role`, `goal`, `brief`, `rules` |
| The current state | `start_turn()` to load it, `describe()` to summarise it |
| How to look at it | read tools: `run(args, turn) -> str` |
| What can be proposed | propose tools: `run(args, turn) -> (proposal, summary)` |
| How an accepted card is applied | `apply()` and `export()`, or the pack's own client |

Every pack also gets the standard read tools: `read_paper`, `list_directory`,
`read_table` and `check_urls`.

The work in a pack is in its propose tools. Each one takes the model's values
as a pydantic model and does what a careful reviewer would do: check them
against the current state, check the links, leave out what failed, and raise
`ToolError` with a message the model can act on when the whole proposal is
wrong. `src/proposer/packs/opentargets/` is the shorter of the two examples.

Run a pack of your own with `--pack your.module:factory`.

## What the Open Targets pack does and does not do

It proposes:

- `copy` tasks (one file each) in an existing or new step, with any new
  version pins their URLs use;
- a new value for an existing version pin, after checking the sources that use
  it at the new value.

It checks that URLs exist (public `gs://` objects included), that `${pins}`
are known, that a source is not already copied, and that a destination is
under `input/` and not already written. It surfaces the comment above a pin
and will not offer a change the config says not to make.

It does not propose other task types (`copy_many`, `explode`, API downloads),
transformation code for the `pts` stage, or orchestration changes, and it
cannot verify objects in private buckets. It says so on the card rather than
guessing.

The config is edited as text, so comments, banners and quoting stay as the
maintainers wrote them, and every edit is verified by parsing the result.

## Settings

| Variable | Meaning | Default |
|---|---|---|
| `PROPOSER_URL` | OpenAI-compatible endpoint, including `/v1` | none (required) |
| `PROPOSER_MODEL` | Model name | `qwen3:14b` |
| `PROPOSER_API_KEY` | Bearer token, if the endpoint needs one | none |
| `PROPOSER_CONTEXT_CHARS` | Prompt budget in characters | `24000` |
| `PROPOSER_TIMEOUT` | Seconds allowed per turn | `600` |
| `PROPOSER_REASONING` | `reasoning_effort` sent by default | `none` |
| `PROPOSER_ALLOW_PRIVATE_URLS` | Let the readers reach private addresses | off |
| `PROPOSER_OPENTARGETS_CONFIG` | Path or URL of `pis/config.yaml` | `main` on GitHub |

## Development

    uv run pytest          # no network or model needed
    uv run ruff check .

The tests drive the engine with a scripted fake model and canned URL checks.
Set `PROPOSER_NETWORK_TESTS=1` to also run the few that read real papers.

## Origin

The engine, the readers and the Tersect pack were extracted from the dataset
assistant of Tersect Browser.
