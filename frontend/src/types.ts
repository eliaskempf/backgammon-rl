// TypeScript mirror of the backend API contract (bgrl/web/schemas.py).
// The board uses absolute coordinates throughout; BAR/OFF are the env sentinels.

export const BAR = -1;
export const OFF = -2;

export type Color = "white" | "black";
export type WinKind = "single" | "gammon" | "backgammon";

export interface CheckerCounts {
  white: number;
  black: number;
}

export interface CubeView {
  value: number;
  owner: Color | null;
}

export interface StateView {
  board: number[]; // length 24, signed: >0 white, <0 black
  bar: CheckerCounts;
  off: CheckerCounts;
  turn: Color;
  cube: CubeView;
}

export interface SubmoveView {
  src: number;
  dst: number;
  die: number | null; // die value this submove consumes (populated on responses)
}

export interface MoveView {
  id: number;
  submoves: SubmoveView[]; // canonical ordering (used for the notation)
  orderings: SubmoveView[][]; // every legal submove ordering reaching this afterstate
  notation: string;
  afterstate: StateView;
}

export interface OutcomeView {
  winner: Color;
  kind: WinKind;
}

export interface CheckpointInfo {
  name: string;
  trained_with: string | null;
  games_trained: number | null;
  created_at: string | null;
  win_rate: number | null;
  eval_opponent: string | null;
}

export interface NewGameResponse {
  game_id: string;
  state: StateView;
  human_color: Color;
  opponent: string;
  to_act: Color;
  needs_roll: boolean;
  manual_dice: boolean;
  can_undo: boolean;
}

export interface RollResponse {
  dice: [number, number];
  to_act: Color;
  auto_pass: boolean;
  n_legal: number;
  state: StateView;
  needs_roll: boolean;
  terminal: boolean;
  outcome: OutcomeView | null;
  can_undo: boolean;
}

export interface LegalMovesResponse {
  dice: [number, number] | null;
  moves: MoveView[];
}

export interface MoveResponse {
  ok: boolean;
  state: StateView;
  to_act: Color;
  needs_roll: boolean;
  terminal: boolean;
  outcome: OutcomeView | null;
  win_prob: number | null; // your win chance after this move (per opponent net)
  can_undo: boolean;
}

export interface AgentMoveResponse {
  move: MoveView | null;
  dice: [number, number];
  state: StateView;
  to_act: Color;
  needs_roll: boolean;
  terminal: boolean;
  outcome: OutcomeView | null;
  win_prob: number | null; // the agent's win chance; UI shows the complement
  can_undo: boolean;
}

export interface UndoResponse {
  state: StateView;
  to_act: Color;
  dice: [number, number] | null;
  needs_roll: boolean;
  terminal: boolean;
  outcome: OutcomeView | null;
  moves: MoveView[];
  can_undo: boolean;
}

export interface CheckpointsResponse {
  checkpoints: CheckpointInfo[];
}

export function otherColor(color: Color): Color {
  return color === "white" ? "black" : "white";
}
