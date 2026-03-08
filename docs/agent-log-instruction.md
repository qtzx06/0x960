# agent log instruction

Use this as a reusable instruction snippet for coding agents working on 0x960.

## short snippet

After each meaningful implementation step, append a short entry to `docs/process.md`.

Each entry should:

- use the current timestamp
- summarize what changed in 2-5 factual bullets
- note any important decisions or blockers
- end with a clear next step

Do not paste large raw command outputs into the log. Summarize them instead.

## longer snippet

You are working in the 0x960 repo. Maintain `docs/process.md` as the project build log.

Rules:

- append to the file after each meaningful work block, not after every micro-step
- keep entries concise and factual
- include what changed, why it changed, blockers, and the next step
- prefer evidence summaries over raw terminal dumps
- optimize the log for demo storytelling and judge review

If you make a product or architecture decision, record it. If a test fails, record the failure briefly and say what remains to fix.
