# Lollm quality path (model-first, non-deterministic)

This repository follows a **model-first** strategy for coding output quality.

## Required behavior

1. Do **not** add deterministic/template shortcuts that bypass LLM generation for user coding prompts.
2. Prefer improvements in:
   - training data quality
   - retrieval relevance
   - prompt construction
   - decoding, scoring, and repair loops
   - model size/architecture/training stability
3. Treat **GPT-5.3-Codex parity** as the quality target direction for code quality, correctness, and reasoning depth.
4. When quality is insufficient, improve the model pipeline and evaluation loop rather than hardcoding endpoint-specific outputs.

## Engineering guidance

- Keep outputs compile-oriented and requirement-aligned.
- Surface rich diagnostics when model calls fail or time out.
- Track quality with repeatable eval prompts and failure-pattern feedback.
- Prefer scalable quality gains (corpus, training profiles, verification, orchestration) over one-off fixes.
