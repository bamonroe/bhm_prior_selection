---
name: article-compute
description: Run this article's computation in containers on the SSH host work. Use when asked to choose an estimation stage, run or smoke-test it remotely, or check, retrieve, or verify its backed-up results.
---

# Article compute: bhm_prior_selection

Read [the shared article compute guide](../../../../article-compute/SKILL-GUIDE.md) first. If this folder is open outside the articles tree, the guide on Pickle is at `~/articles/article-compute/SKILL-GUIDE.md`.

Read `code/main.py` and `runners/article_smoke.py`. The tested recipe is `article-run submit bhm-prior-smoke`: one CRRA + Prelec posterior stage on CUDA, 8 chains per group and 12 draws per chain, with the already registered `bhm-prior-smoke-v1` inputs. Check for both a checkpoint and a history. The short chains are not intended to converge.
