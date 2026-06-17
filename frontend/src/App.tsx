import { useCallback, useEffect, useMemo, useState } from "react";

import { ApiError, api } from "./api";
import Board from "./Board";
import {
  type CheckpointInfo,
  type Color,
  type MoveView,
  type OutcomeView,
  type StateView,
  type SubmoveView,
  otherColor,
} from "./types";

function isPrefix(chosen: SubmoveView[], full: SubmoveView[]): boolean {
  return chosen.every((sm, i) => full[i] && full[i].src === sm.src && full[i].dst === sm.dst);
}

function sameSubmoves(a: SubmoveView[], b: SubmoveView[]): boolean {
  return a.length === b.length && isPrefix(a, b);
}

export default function App() {
  // New-game form.
  const [formColor, setFormColor] = useState<Color>("white");
  const [formOpponent, setFormOpponent] = useState("random");
  const [formSeed, setFormSeed] = useState("");
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);

  // Live game.
  const [gameId, setGameId] = useState<string | null>(null);
  const [humanColor, setHumanColor] = useState<Color>("white");
  const [view, setView] = useState<StateView | null>(null);
  const [toAct, setToAct] = useState<Color>("white");
  const [needsRoll, setNeedsRoll] = useState(false);
  const [terminal, setTerminal] = useState(false);
  const [outcome, setOutcome] = useState<OutcomeView | null>(null);
  const [dice, setDice] = useState<[number, number] | null>(null);

  // Human move construction.
  const [candidates, setCandidates] = useState<MoveView[]>([]);
  const [chosen, setChosen] = useState<SubmoveView[]>([]);
  const [selected, setSelected] = useState<number | null>(null);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastAgentMove, setLastAgentMove] = useState<MoveView | null>(null);

  useEffect(() => {
    api
      .checkpoints()
      .then((r) => setCheckpoints(r.checkpoints))
      .catch(() => setCheckpoints([]));
  }, []);

  const applyResult = useCallback(
    (r: { state: StateView; to_act: Color; needs_roll: boolean; terminal: boolean; outcome: OutcomeView | null }) => {
      setView(r.state);
      setToAct(r.to_act);
      setNeedsRoll(r.needs_roll);
      setTerminal(r.terminal);
      setOutcome(r.outcome);
    },
    [],
  );

  // Let the opponent play until it is the human's turn (or the game ends).
  const runAgent = useCallback(
    async (gid: string, startTurn: Color, startTerminal: boolean, oppColor: Color) => {
      let turn = startTurn;
      let done = startTerminal;
      while (!done && turn === oppColor) {
        const a = await api.agentMove(gid);
        setLastAgentMove(a.move);
        setDice(a.dice);
        applyResult(a);
        turn = a.to_act;
        done = a.terminal;
      }
    },
    [applyResult],
  );

  const newGame = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const seed = formSeed.trim() === "" ? null : Number(formSeed);
      const ng = await api.newGame({ human_color: formColor, opponent: formOpponent, seed });
      setGameId(ng.game_id);
      setHumanColor(ng.human_color);
      setView(ng.state);
      setToAct(ng.to_act);
      setNeedsRoll(ng.needs_roll);
      setTerminal(false);
      setOutcome(null);
      setDice(null);
      setCandidates([]);
      setChosen([]);
      setSelected(null);
      setLastAgentMove(null);
      await runAgent(ng.game_id, ng.to_act, false, otherColor(ng.human_color));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [formColor, formOpponent, formSeed, runAgent]);

  const roll = useCallback(async () => {
    if (!gameId) return;
    setBusy(true);
    setError(null);
    try {
      const r = await api.roll(gameId);
      setDice(r.dice);
      applyResult(r);
      if (r.auto_pass) {
        setCandidates([]);
        setChosen([]);
        setSelected(null);
        await runAgent(gameId, r.to_act, r.terminal, otherColor(humanColor));
      } else {
        const lm = await api.legalMoves(gameId);
        setCandidates(lm.moves);
        setChosen([]);
        setSelected(null);
      }
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [gameId, humanColor, applyResult, runAgent]);

  const submitMove = useCallback(
    async (moveId: number) => {
      if (!gameId) return;
      setBusy(true);
      setError(null);
      try {
        const r = await api.move(gameId, moveId);
        applyResult(r);
        setCandidates([]);
        setChosen([]);
        setSelected(null);
        setDice(null);
        await runAgent(gameId, r.to_act, r.terminal, otherColor(humanColor));
      } catch (e) {
        setError(e instanceof ApiError ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [gameId, humanColor, applyResult, runAgent],
  );

  // Derived sets that drive board highlighting (recomputed from the chosen prefix).
  const matching = useMemo(
    () => candidates.filter((c) => isPrefix(chosen, c.submoves)),
    [candidates, chosen],
  );
  const sources = useMemo(() => {
    const s = new Set<number>();
    for (const c of matching) {
      const next = c.submoves[chosen.length];
      if (next) s.add(next.src);
    }
    return s;
  }, [matching, chosen.length]);
  const destinations = useMemo(() => {
    const d = new Set<number>();
    if (selected === null) return d;
    for (const c of matching) {
      const next = c.submoves[chosen.length];
      if (next && next.src === selected) d.add(next.dst);
    }
    return d;
  }, [matching, chosen.length, selected]);

  const humanTurn = !!gameId && !terminal && toAct === humanColor && !busy;

  const onPick = useCallback(
    (square: number) => {
      if (!humanTurn || !dice) return;
      if (selected === null) {
        if (sources.has(square)) setSelected(square);
        return;
      }
      if (destinations.has(square)) {
        const next = [...chosen, { src: selected, dst: square }];
        setSelected(null);
        const complete = candidates.find((c) => sameSubmoves(c.submoves, next));
        if (complete) {
          void submitMove(complete.id);
        } else {
          setChosen(next);
        }
        return;
      }
      // Clicking another source re-selects; anything else clears the selection.
      setSelected(sources.has(square) ? square : null);
    },
    [humanTurn, dice, selected, sources, destinations, chosen, candidates, submitMove],
  );

  const undo = useCallback(() => {
    setSelected(null);
    setChosen((c) => c.slice(0, -1));
  }, []);

  const statusLine = () => {
    if (!gameId) return "Start a new game.";
    if (terminal && outcome) {
      const who = outcome.winner === humanColor ? "You win" : "Opponent wins";
      return `Game over — ${who} (${outcome.kind}).`;
    }
    if (toAct === humanColor) return needsRoll ? "Your turn — roll the dice." : "Your turn — choose a move.";
    return "Opponent is thinking…";
  };

  return (
    <div className="app">
      <h1>bgrl — play backgammon</h1>

      <div className="controls">
        <label>
          Play as{" "}
          <select value={formColor} onChange={(e) => setFormColor(e.target.value as Color)}>
            <option value="white">white</option>
            <option value="black">black</option>
          </select>
        </label>
        <label>
          Opponent{" "}
          <select value={formOpponent} onChange={(e) => setFormOpponent(e.target.value)}>
            <option value="random">random</option>
            {checkpoints.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name}
                {c.games_trained != null ? ` (${c.games_trained} games)` : ""}
              </option>
            ))}
          </select>
        </label>
        <label>
          Seed{" "}
          <input
            type="text"
            value={formSeed}
            placeholder="optional"
            size={6}
            onChange={(e) => setFormSeed(e.target.value)}
          />
        </label>
        <button onClick={newGame} disabled={busy}>
          New game
        </button>
      </div>

      <div className="status">{statusLine()}</div>
      {error && <div className="error">⚠ {error}</div>}

      {view && (
        <Board
          state={view}
          sources={humanTurn ? sources : new Set()}
          destinations={humanTurn ? destinations : new Set()}
          selected={selected}
          dice={dice}
          onPick={onPick}
        />
      )}

      {gameId && !terminal && (
        <div className="actionbar">
          {toAct === humanColor && needsRoll && (
            <button onClick={roll} disabled={busy}>
              Roll
            </button>
          )}
          {humanTurn && !needsRoll && chosen.length > 0 && (
            <button onClick={undo} disabled={busy}>
              Undo
            </button>
          )}
          {humanTurn && !needsRoll && candidates.length > 0 && (
            <div className="legal">
              <span>Legal moves:</span>
              {candidates.map((m) => (
                <button key={m.id} onClick={() => submitMove(m.id)} disabled={busy}>
                  {m.notation}
                </button>
              ))}
            </div>
          )}
          {lastAgentMove && (
            <div className="agentmove">Opponent played: {lastAgentMove.notation}</div>
          )}
        </div>
      )}
    </div>
  );
}
