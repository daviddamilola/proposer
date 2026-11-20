# Changelog

## 0.2.0

- The Claude API as a second backend (`proposer.claude`, the `claude` extra):
  `--claude` on the command line, and a `backend` argument to
  `engine.run_turn`, `server.event_stream`, `server.router` and
  `server.create_app`. The default is unchanged: an OpenAI-compatible
  endpoint.
- The `tersect` pack is no longer part of the package: it lives in Tersect
  Browser's repository, which maintains it. Replace
  `from proposer.packs import tersect` with your own pack; the command line
  loads any installed pack by `module:function`. A frozen copy remains in
  `tests/dataset_pack`, as the pack the engine is tested through.
- `stream_chat` takes `final`: on the last round of a turn the backend, not
  the engine, decides how to keep the model from calling tools.

## 0.1.0

First release.

- The engine: a bounded tool-calling loop in which a model may only read and
  propose, against any OpenAI-compatible endpoint.
- The pack interface, with two packs: `tersect` and `opentargets`.
- Skills pinned by a pack, and ToolUniverse as an optional source of read
  tools and skills (`proposer-llm[tooluniverse]`).
- `proposer chat` and `proposer serve`.
- Settings under an embedding application's own variable names (`settings.alias`).
