# Architectures

`pe.pack(model)` finds the backbone inside whatever it is handed (a Hugging Face model, a
`SentenceTransformer`, a PyLate `ColBERT`) and asks each registered architecture whether it
patches that module. The first match installs its engine in place. `unpack()`, `validate()`,
`set_cuda_graph()` and `no_cuda_graph()` route the same way.

| name         | backbone                                                   | patched entry point        |
|--------------|------------------------------------------------------------|----------------------------|
| `modernbert` | ModernBERT, Ettin, mmBERT and their finetunes              | `ModernBertModel.forward`  |

Each architecture brings its own kernel toolchain, loaded only when a model of that
architecture is packed: CuteDSL for ModernBERT. `import packed_encoders` itself pulls in none.

## Adding an architecture

1. Implement the `Architecture` protocol (`arch/base.py`) in `arch/<name>/`:
   `match(module)` (exact: two plugins must never both match), `validate`, `pack`, `unpack`.
2. Put the math in an engine that replaces the module's `forward` with the same function in
   fewer kernels, and gate it: `validate` compares it with the module's own forward on the GPU
   in hand and raises `ValidationError` on a miss. The state it stores on the module exposes
   `graph_enabled` and `set_cuda_graph(enabled, config)`.
3. Import the toolchain lazily (inside `pack`/`validate`), so `import packed_encoders` stays
   free of it, and `register()` the plugin in `arch/__init__.py`.
