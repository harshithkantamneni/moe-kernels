# Rental 1 scorers (2026-10-01)

The scripts that scored rental 1's four registrations
(`results/published/2026-10-01-nvidia_gh200_480gb-rental1-session`), with their outputs
(`<name>.score.json`, `<name>.score.txt`). Each reads the registered files exactly as
committed at 436e41c and the published pages; nothing in them is fitted on the pages.

    python scripts/scoring/rental1/score_floor.py <repo> <session dir or run tree> <out dir>

(the same three arguments for each; see the top of each script.)

| scorer | registration | written |
|---|---|---|
| score_floor.py | 2026-09-30 mixtral-8x7b-tp8 floor | after the pages, from the registered JSON's per-cell predictions |
| score_launch.py | 2026-10-01 launch floor (A) | after the pages: no scorer was registered; P6's max-plus model is its reading of the registered text |
| score_l2.py | 2026-10-01 L2 survival, TP shards (B) | after the pages, on the design's registered law and bands |
| score_atile.py | 2026-10-01 A-tile k-steps (C) | after the pages, on the registered fits and bands |

All four were written after the pages existed; the registrations fixed the numbers and the
bands, the scorers only apply them. A scorer registered before its pages is the rule from
rental 2 on.

**Fixed 2026-10-02 (print only).** `score_floor.py` l.40 read a gate's `id` or `name`; the
floor captures key their gates `number`, so `floor.score.txt` printed `gates [(None,
'FAIL')]`. It now reads `number` first; `floor.score.{json,txt}` were regenerated and differ
only in those two labels (now `FL1`). No number or verdict changed.

**Moved 2026-10-05 (no output changed).** `atile.score.txt` carried 17 hand-written verdict
lines at its top and `l2.score.txt` 12 at its end, which no rerun prints. They are now in
`NOTES.md` beside this file, unchanged, and both `.score.txt` files are the scorers' own output
again (a rerun reproduces all eight committed outputs byte for byte). T1 on Qwen2-57B is the
scorer's NOT HELD; the hand note's "Neither pass nor fail" says the same, and the registration
defines no INCONCLUSIVE for T1.
