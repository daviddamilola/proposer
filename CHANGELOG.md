# Changelog

## 0.1.0

First release.

- The engine: a bounded tool-calling loop in which a model may only read and
  propose, against any OpenAI-compatible endpoint.
- The pack interface, with two packs: `tersect` and `opentargets`.
- Skills pinned by a pack, and ToolUniverse as an optional source of read
  tools and skills (`proposer-llm[tooluniverse]`).
- `proposer chat` and `proposer serve`.
- Settings under an embedding application's own variable names (`settings.alias`).
