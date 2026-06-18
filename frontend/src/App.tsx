import { useCallback, useEffect, useMemo, useState } from "react";

import { ApiError, api } from "./api";
import Board, { type GlideAnim } from "./Board";
import { applySubmoves } from "./game";
import {
  type CheckpointInfo,
  type Color,
  type MoveView,
  type OutcomeView,
  type StateView,
  type SubmoveView,
  otherColor,
} from "./types";

// Animation timing (ms). Tuned to read clearly without dragging; tweak freely.
const AGENT_REVEAL_MS = 700; // pause after showing the opponent's dice, before it moves
const AGENT_GLIDE_MS = 450; // checker slide duration for the opponent
const AGENT_SETTLE_MS = 250; // glow/settle after each opponent submove
const HUMAN_GLIDE_MS = 240; // snappier slide for the human's own submoves
const HUMAN_SETTLE_MS = 100;
const REPLAY_REVEAL_MS = 350;

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

function isPrefix(chosen: SubmoveView[], full: SubmoveView[]): boolean {
  return chosen.every((sm, i) => full[i] && full[i].src === sm.src && full[i].dst === sm.dst);
}

function sameSubmoves(a: SubmoveView[], b: SubmoveView[]): boolean {
  return a.length === b.length && isPrefix(a, b);
}

// Display order: higher die on the left by default; the flip toggle reverses it.
function orderDice(dice: [number, number] | null, swapped: boolean): [number, number] | null {
  if (!dice) return null;
  const hi = Math.max(dice[0], dice[1]);
  const lo = Math.min(dice[0], dice[1]);
  return swapped ? [lo, hi] : [hi, lo];
}

// Parenthetical shown next to a checkpoint in the opponent picker: training volume and,
// when recorded, its eval win rate (e.g. "300000 games, 65% vs pubeval").
function opponentSuffix(c: CheckpointInfo): string {
  const parts: string[] = [];
  if (c.games_trained != null) parts.push(`${c.games_trained} games`);
  if (c.win_rate != null) {
    parts.push(`${Math.round(c.win_rate * 100)}% vs ${c.eval_opponent ?? "baseline"}`);
  }
  return parts.length ? ` (${parts.join(", ")})` : "";
}

// Manual-dice input: two 1..6 fields the human fills to roll for either seat. Submit is
// disabled until both are valid; the entry self-clears after a roll so the next is fresh.
function DiceEntry({
  label,
  busy,
  onSubmit,
}: {
  label: string;
  busy: boolean;
  onSubmit: (dice: [number, number]) => void;
}) {
  const [d0, setD0] = useState("");
  const [d1, setD1] = useState("");
  const a = Number(d0);
  const b = Number(d1);
  const valid = [d0, d1].every((s) => s !== "") && [a, b].every((n) => Number.isInteger(n) && n >= 1 && n <= 6);
  const submit = () => {
    if (!valid || busy) return;
    onSubmit([a, b]);
    setD0("");
    setD1("");
  };
  return (
    <div className="diceentry">
      <span>{label}</span>
      {[
        [d0, setD0] as const,
        [d1, setD1] as const,
      ].map(([val, set], i) => (
        <input
          key={i}
          type="number"
          min={1}
          max={6}
          value={val}
          placeholder="?"
          onChange={(e) => set(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
        />
      ))}
      <button onClick={submit} disabled={!valid || busy}>
        Set dice
      </button>
    </div>
  );
}

interface ReplayData {
  fromBoard: StateView;
  submoves: SubmoveView[];
  mover: Color;
  dice: [number, number];
}

export default function App() {
  // New-game form.
  const [formColor, setFormColor] = useState<Color>("white");
  const [formOpponent, setFormOpponent] = useState("random");
  const [formSeed, setFormSeed] = useState("");
  const [formManual, setFormManual] = useState(false);
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);

  // Live game.
  const [gameId, setGameId] = useState<string | null>(null);
  const [humanColor, setHumanColor] = useState<Color>("white");
  const [opponentName, setOpponentName] = useState("random");
  const [manual, setManual] = useState(false);
  const [view, setView] = useState<StateView | null>(null);
  const [toAct, setToAct] = useState<Color>("white");
  const [needsRoll, setNeedsRoll] = useState(false);
  const [terminal, setTerminal] = useState(false);
  const [outcome, setOutcome] = useState<OutcomeView | null>(null);
  const [dice, setDice] = useState<[number, number] | null>(null);
  const [diceSwapped, setDiceSwapped] = useState(false);

  // Human move construction.
  const [candidates, setCandidates] = useState<MoveView[]>([]);
  const [chosen, setChosen] = useState<SubmoveView[]>([]);

  // Animation, value readout, undo, replay.
  const [animation, setAnimation] = useState<GlideAnim | null>(null);
  const [winChance, setWinChance] = useState<number | null>(null); // human POV (0..1)
  const [canUndo, setCanUndo] = useState(false);
  const [replay, setReplay] = useState<ReplayData | null>(null);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastAgentMove, setLastAgentMove] = useState<MoveView | null>(null);

  useEffect(() => {
    api
      .checkpoints()
      // Order the picker by training volume so a difficulty ladder reads weak -> strong.
      .then((r) =>
        setCheckpoints(
          [...r.checkpoints].sort((a, b) => (a.games_trained ?? 0) - (b.games_trained ?? 0)),
        ),
      )
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

  // Run a declarative checker glide on the Board and resolve when it finishes.
  const animate = useCallback(
    (spec: Omit<GlideAnim, "onDone">) =>
      new Promise<void>((resolve) => {
        setAnimation({
          ...spec,
          onDone: () => {
            setAnimation(null);
            resolve();
          },
        });
      }),
    [],
  );

  // Let the opponent play (animating each move) until it is the human's turn or the
  // game ends. ``startBoard`` is the board the first opponent move animates from.
  const runAgent = useCallback(
    async (gid: string, startTurn: Color, startTerminal: boolean, oppColor: Color, startBoard: StateView) => {
      let turn = startTurn;
      let done = startTerminal;
      let board = startBoard;
      while (!done && turn === oppColor) {
        const a = await api.agentMove(gid);
        setLastAgentMove(a.move);
        setDice(a.dice); // reveal the opponent's roll first
        if (a.move && a.move.submoves.length) {
          await animate({
            fromBoard: board,
            submoves: a.move.submoves,
            mover: oppColor,
            startDelayMs: AGENT_REVEAL_MS,
            glideMs: AGENT_GLIDE_MS,
            settleMs: AGENT_SETTLE_MS,
          });
          setReplay({ fromBoard: board, submoves: a.move.submoves, mover: oppColor, dice: a.dice });
        } else {
          await sleep(AGENT_REVEAL_MS); // forced pass: just a beat
          setReplay(null);
        }
        applyResult(a);
        if (a.win_prob != null) setWinChance(1 - a.win_prob); // back to "your" win chance
        setCanUndo(a.can_undo);
        board = a.state;
        turn = a.to_act;
        done = a.terminal;
      }
      if (!done) setDice(null); // human's turn to roll; clear the opponent's dice
    },
    [animate, applyResult],
  );

  const newGame = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const seed = formSeed.trim() === "" ? null : Number(formSeed);
      const ng = await api.newGame({
        human_color: formColor,
        opponent: formOpponent,
        seed,
        manual_dice: formManual,
      });
      setGameId(ng.game_id);
      setHumanColor(ng.human_color);
      setOpponentName(ng.opponent);
      setManual(ng.manual_dice);
      setView(ng.state);
      setToAct(ng.to_act);
      setNeedsRoll(ng.needs_roll);
      setTerminal(false);
      setOutcome(null);
      setDice(null);
      setDiceSwapped(false);
      setCandidates([]);
      setChosen([]);
      setLastAgentMove(null);
      setWinChance(null);
      setCanUndo(ng.can_undo);
      setReplay(null);
      // In manual mode the human drives the opponent's rolls too, so don't auto-play;
      // the action bar surfaces the opponent dice entry when it's the opponent's turn.
      if (!ng.manual_dice) {
        await runAgent(ng.game_id, ng.to_act, false, otherColor(ng.human_color), ng.state);
      }
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [formColor, formOpponent, formSeed, formManual, runAgent]);

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
        setDice(null);
        if (r.win_prob != null) setWinChance(r.win_prob); // your win chance after the move
        setCanUndo(r.can_undo);
        if (!manual) await runAgent(gameId, r.to_act, r.terminal, otherColor(humanColor), r.state);
      } catch (e) {
        setError(e instanceof ApiError ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [gameId, humanColor, manual, applyResult, runAgent],
  );

  const roll = useCallback(
    async (suppliedDice?: [number, number]) => {
      if (!gameId) return;
      setBusy(true);
      setError(null);
      try {
        const r = await api.roll(gameId, suppliedDice);
        setDice(r.dice);
        setDiceSwapped(false);
        setCanUndo(r.can_undo);
        applyResult(r);
        if (r.auto_pass) {
          setCandidates([]);
          setChosen([]);
          if (!manual) await runAgent(gameId, r.to_act, r.terminal, otherColor(humanColor), r.state);
        } else {
          const lm = await api.legalMoves(gameId);
          setCandidates(lm.moves);
          setChosen([]);
        }
      } catch (e) {
        setError(e instanceof ApiError ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [gameId, humanColor, manual, applyResult, runAgent],
  );

  // Manual mode: play one opponent ply using the human-supplied dice, animated.
  const manualAgentMove = useCallback(
    async (suppliedDice: [number, number]) => {
      if (!gameId || !view) return;
      setBusy(true);
      setError(null);
      try {
        const board = view;
        const a = await api.agentMove(gameId, suppliedDice);
        setLastAgentMove(a.move);
        setDice(a.dice);
        if (a.move && a.move.submoves.length) {
          await animate({
            fromBoard: board,
            submoves: a.move.submoves,
            mover: otherColor(humanColor),
            startDelayMs: AGENT_REVEAL_MS,
            glideMs: AGENT_GLIDE_MS,
            settleMs: AGENT_SETTLE_MS,
          });
          setReplay({ fromBoard: board, submoves: a.move.submoves, mover: otherColor(humanColor), dice: a.dice });
        } else {
          await sleep(AGENT_REVEAL_MS);
          setReplay(null);
        }
        applyResult(a);
        if (a.win_prob != null) setWinChance(1 - a.win_prob);
        setCanUndo(a.can_undo);
        setDice(null);
      } catch (e) {
        setError(e instanceof ApiError ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [gameId, view, humanColor, animate, applyResult],
  );

  // Derived: matching candidates for the chosen prefix, and the clickable sources.
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

  const humanTurn = !!gameId && !terminal && toAct === humanColor && !busy && !animation;
  const displayedDice = useMemo(() => orderDice(dice, diceSwapped), [dice, diceSwapped]);

  // The optimistic preview board: the server state with the human's chosen submoves
  // applied, so each played submove is visible before the full move is submitted.
  const preview = useMemo(
    () => (view && toAct === humanColor ? applySubmoves(view, chosen, humanColor) : view),
    [view, chosen, humanColor, toAct],
  );

  const onPick = useCallback(
    async (square: number) => {
      if (!humanTurn || !displayedDice || animation || !preview) return;
      const nexts = matching
        .map((c) => c.submoves[chosen.length])
        .filter((sm): sm is SubmoveView => !!sm && sm.src === square);
      if (!nexts.length) return;
      const [leftDie, rightDie] = displayedDice;
      const pick =
        nexts.find((sm) => sm.die === leftDie) ?? nexts.find((sm) => sm.die === rightDie) ?? nexts[0];
      const oldPreview = preview;
      const newChosen = [...chosen, pick];
      const complete = candidates.find((c) => sameSubmoves(c.submoves, newChosen));
      setChosen(newChosen);
      await animate({
        fromBoard: oldPreview,
        submoves: [pick],
        mover: humanColor,
        startDelayMs: 0,
        glideMs: HUMAN_GLIDE_MS,
        settleMs: HUMAN_SETTLE_MS,
      });
      if (complete) await submitMove(complete.id);
    },
    [humanTurn, displayedDice, animation, preview, matching, chosen, candidates, humanColor, animate, submitMove],
  );

  // Fallback: play a full legal move from the notation list, animating what's left.
  const playFullMove = useCallback(
    async (m: MoveView) => {
      if (animation || !preview) return;
      const remaining = m.submoves.slice(chosen.length);
      const oldPreview = preview;
      setChosen(m.submoves);
      if (remaining.length) {
        await animate({
          fromBoard: oldPreview,
          submoves: remaining,
          mover: humanColor,
          startDelayMs: 0,
          glideMs: HUMAN_GLIDE_MS,
          settleMs: HUMAN_SETTLE_MS,
        });
      }
      await submitMove(m.id);
    },
    [animation, preview, chosen.length, humanColor, animate, submitMove],
  );

  const onUndo = useCallback(async () => {
    if (chosen.length > 0) {
      setChosen((c) => c.slice(0, -1)); // step back within the current move
      return;
    }
    if (!gameId || !canUndo) return;
    setBusy(true);
    setError(null);
    try {
      const r = await api.undo(gameId);
      setView(r.state);
      setToAct(r.to_act);
      setNeedsRoll(r.needs_roll);
      setTerminal(r.terminal);
      setOutcome(r.outcome);
      setDice(r.dice);
      setDiceSwapped(false);
      setCandidates(r.moves);
      setChosen([]);
      setLastAgentMove(null);
      setWinChance(null);
      setReplay(null);
      setCanUndo(r.can_undo);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [chosen.length, gameId, canUndo]);

  const onReplay = useCallback(async () => {
    if (!replay || animation) return;
    const prev = dice;
    setDice(replay.dice);
    await animate({
      fromBoard: replay.fromBoard,
      submoves: replay.submoves,
      mover: replay.mover,
      startDelayMs: REPLAY_REVEAL_MS,
      glideMs: AGENT_GLIDE_MS,
      settleMs: AGENT_SETTLE_MS,
    });
    setDice(prev);
  }, [replay, animation, dice, animate]);

  // Dice presentation: colour for whoever the dice belong to, plus "spent"/"remaining"
  // indicators for the human as they consume each die.
  const diceColor: Color = animation
    ? animation.mover
    : toAct === humanColor
      ? humanColor
      : otherColor(humanColor);
  const humanChoosing = !!gameId && !terminal && toAct === humanColor && !needsRoll;
  const isDouble = !!dice && dice[0] === dice[1];
  const consumed = humanChoosing ? chosen.map((c) => c.die ?? 0) : [];
  const diceSpent: [boolean, boolean] =
    humanChoosing && displayedDice && !isDouble
      ? [consumed.includes(displayedDice[0]), consumed.includes(displayedDice[1])]
      : [isDouble && humanChoosing && consumed.length >= 4, isDouble && humanChoosing && consumed.length >= 4];
  const diceRemaining = humanChoosing && isDouble ? Math.max(4 - consumed.length, 0) : null;
  const glowPoint = chosen.length > 0 ? chosen[chosen.length - 1].dst : null;
  const canFlip = humanChoosing && !animation && !!dice && !isDouble;

  const statusLine = () => {
    if (!gameId) return "Start a new game.";
    if (terminal && outcome) {
      const who = outcome.winner === humanColor ? "You win" : "Opponent wins";
      return `Game over — ${who} (${outcome.kind}).`;
    }
    if (toAct === humanColor) {
      if (!needsRoll) return "Your turn — click a checker to move it.";
      return manual ? "Your turn — enter your dice." : "Your turn — roll the dice.";
    }
    return manual ? "Enter the opponent's dice to roll for it." : "Opponent is thinking…";
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
                {opponentSuffix(c)}
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
        <label title="You enter every roll — yours and the opponent's — so the bot never draws its own dice.">
          <input
            type="checkbox"
            checked={formManual}
            onChange={(e) => setFormManual(e.target.checked)}
          />{" "}
          Enter dice manually
        </label>
        <button onClick={newGame} disabled={busy}>
          New game
        </button>
      </div>

      <div className="status">{statusLine()}</div>
      {winChance !== null && (
        <div className="winchance">
          Your win chance: <strong>{Math.round(winChance * 100)}%</strong>{" "}
          <span className="muted">(estimated by the {opponentName} net)</span>
        </div>
      )}
      {error && <div className="error">⚠ {error}</div>}

      {view && (
        <Board
          state={preview ?? view}
          sources={humanTurn ? sources : new Set()}
          glowPoint={glowPoint}
          dice={displayedDice}
          diceColor={diceColor}
          diceSpent={diceSpent}
          diceRemaining={diceRemaining}
          animation={animation}
          onPick={onPick}
          onFlipDice={canFlip ? () => setDiceSwapped((s) => !s) : undefined}
        />
      )}

      {gameId && !terminal && (
        <div className="actionbar">
          {toAct === humanColor && needsRoll && !manual && (
            <button onClick={() => roll()} disabled={busy || !!animation}>
              Roll
            </button>
          )}
          {toAct === humanColor && needsRoll && manual && (
            <DiceEntry label="Your dice:" busy={busy} onSubmit={(d) => roll(d)} />
          )}
          {toAct !== humanColor && manual && !animation && (
            <DiceEntry label="Opponent's dice:" busy={busy} onSubmit={manualAgentMove} />
          )}
          {(chosen.length > 0 || (toAct === humanColor && canUndo)) && (
            <button onClick={onUndo} disabled={busy || !!animation}>
              {chosen.length > 0 ? "Undo move" : "Undo turn"}
            </button>
          )}
          {replay && (
            <button onClick={onReplay} disabled={busy || !!animation}>
              Replay opponent move
            </button>
          )}
          {humanTurn && !needsRoll && matching.length > 0 && (
            <div className="legal">
              <span>Legal moves:</span>
              {matching.map((m) => (
                <button key={m.id} onClick={() => playFullMove(m)} disabled={busy || !!animation}>
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
