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
}

export interface MoveView {
  id: number;
  submoves: SubmoveView[];
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
}

export interface NewGameResponse {
  game_id: string;
  state: StateView;
  human_color: Color;
  opponent: string;
  to_act: Color;
  needs_roll: boolean;
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
}

export interface AgentMoveResponse {
  move: MoveView | null;
  dice: [number, number];
  state: StateView;
  to_act: Color;
  needs_roll: boolean;
  terminal: boolean;
  outcome: OutcomeView | null;
}

export interface CheckpointsResponse {
  checkpoints: CheckpointInfo[];
}

export function otherColor(color: Color): Color {
  return color === "white" ? "black" : "white";
}
