# Why Chess960

## short version

Chess960 keeps the rules of chess fixed while randomizing the starting position across 960 legal setups. That makes it a clean distribution-shift benchmark: the task is still chess, but many standard opening-pattern shortcuts become much less useful.

For this project, that matters because we do not want to train against a benchmark that can be improved mostly through memorized opening structure. We want a downstream task where better performance is more likely to reflect transferable evaluation and search behavior.

## why it matters for 0x960

- the environment is still grounded in a familiar domain with clear win/loss signals
- the benchmark is less vulnerable to opening-book memorization than standard chess
- the engine must perform across many starting setups, not just one canonical opening tree
- reward comes from real match outcomes, not proxy text metrics

## what we should claim

We should not claim that Chess960 proves genuine understanding.

We should claim something narrower and more defensible:

- Chess960 is a cleaner robustness test than standard chess alone
- strong standard-chess performance does not automatically transfer
- this makes Chess960 a good downstream benchmark for a tool-using self-improvement environment

## Relation to 0x960

0x960 is not a move-prediction benchmark. The model does not play moves directly as its primary task.

Instead, the model acts like a bounded engine engineer:

- inspect engine files
- edit the eval function
- run checks
- get rewarded by whether the modified engine performs better

That is the key bridge from the research motivation to the OpenEnv environment design.
