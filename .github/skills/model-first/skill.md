# Model-first coding policy skill

## Goal
Keep code generation **LLM-driven** and avoid deterministic/template shortcuts.

## Hard rules
1. Never bypass model generation for user coding prompts with hardcoded route/template outputs.
2. If quality is low, improve retrieval, prompting, scoring, repair loops, and training data.
3. Preserve detailed diagnostics for model/backend failures.
4. Target GPT-5.3-Codex-level coding quality directionally (correctness, reasoning, requirement fit).

## Required response strategy
1. Try model generation first.
2. If output quality is poor, iterate with model-based retries and tighter constraints.
3. Use verification feedback to guide model repair prompts.
4. Do not return deterministic endpoint-specific fallback code.
