# WP7 — Match Play: Cube to 7/11 Points + Match Equity + Crawford

**Status:** not started · **Depends on:** WP6 (5-head net + money cube) + WP3 (`.mat`/`.sgf` export it extends) · **Parallel with:** —

**Branch + worktree:** this session creates branch `wp7-match-play` off the latest `main`
**in its own git worktree** and works only there (CLAUDE.md §10); it never asks the human to
create, switch, merge, or share worktrees, and never edits another WP's worktree.

The split-off from WP6: take the cube-capable money agent and add the **match score layer**.
The whole point of WP6's cube-ready factoring is that **the value net does not change here** —
all score dependence lives in the equity/MET reduction and the cube/match rules.

---

## Scope

- **Match equity table (MET).** Ship a standard table (e.g. Kazaross-XG2 or gnubg's
  G11); do **not** learn it from match outcomes. Add a match-equity path to
  `bgrl/nets/equity.py`: reduce the cubeless 5-vector to **match-winning chance (MWC)** /
  centered match equity using `MET(score after each outcome) − MET(current score)`.
- **Match score in context.** Extend `CubeContext` (or a wrapping match context) with
  `(my_away, opp_away)` and the Crawford flag. **Keep the net score-independent** — it
  predicts the cubeless game-outcome distribution, which does not depend on the score; only
  the reduction sees the score. Do not feed match score into the net's inputs.
- **Crawford + post-Crawford rules** in the game/episode logic: no doubling in the Crawford
  game; free-drop behaviour after. These change legal cube actions and the MET lookup.
- **Match-length episodes.** A match = first to N points (7, 11), composed of multiple
  games; `episode == single game` is finally relaxed (CLAUDE.md §5 anticipated this). The
  RL **target stays the per-game 5-vector** — do **not** retrain on the sparse, high-variance
  match-win signal; the MET handles match-level weighting analytically.
- **Score-aware cube decisions.** Take points and gammon value shift with the score (a
  gammon can be match-winning at one score, worthless at another). Reuse WP6's cube surface;
  swap money equity for match equity in the double/take/pass windows.
- **Export.** Extend WP3's `.mat`/`.sgf` writer to record cube actions, cube value, and
  match score across a full match (the formats are natively match-and-cube shaped; v1 wrote
  1-point "matches"). One match per `.mat` file (gnubg reads only the first).

## Acceptance criteria

- Match equities and score-aware double/take/pass decisions **match gnubg** on a reference
  set across several scores (including Crawford / post-Crawford and double-match-point).
- A full N-point match plays end to end, exports to `.mat`, and gnubg imports + analyses it
  without error.
- The WP6 value net is reused **unchanged** (no retraining for match play); if match play
  seems to need a new net, the factoring was wrong — escalate.
- `uv run pytest` green, `uv run ruff check` clean.

## Notes

- The net being score-independent is the single most important invariant — it is why one
  trained net serves cubeless money, cubeful money (WP6), and any match length (WP7). Pin a
  test that the encoded features do not depend on match score.
- Cube errors are among the largest equity blunders, and cube handling is largely separable
  — analytic Janowski/MET decisions on top of WP6's gammon-aware net are the pragmatic path;
  do not try to learn the cube from scratch with RL.
