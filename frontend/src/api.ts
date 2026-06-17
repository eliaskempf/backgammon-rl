// Thin typed wrapper over the play-server REST API. No game logic lives here.

import type {
  AgentMoveResponse,
  CheckpointsResponse,
  Color,
  LegalMovesResponse,
  MoveResponse,
  NewGameResponse,
  RollResponse,
} from "./types";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function detail(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (body && typeof body.detail === "string") return body.detail;
  } catch {
    /* fall through to status text */
  }
  return res.statusText || `HTTP ${res.status}`;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new ApiError(res.status, await detail(res));
  return (await res.json()) as T;
}

async function get<T>(path: string, params?: Record<string, string>): Promise<T> {
  const qs = params ? "?" + new URLSearchParams(params).toString() : "";
  const res = await fetch(path + qs);
  if (!res.ok) throw new ApiError(res.status, await detail(res));
  return (await res.json()) as T;
}

export const api = {
  newGame: (req: { human_color: Color; opponent: string; seed: number | null }) =>
    post<NewGameResponse>("/new_game", req),
  roll: (gameId: string) => post<RollResponse>("/roll", { game_id: gameId }),
  legalMoves: (gameId: string) =>
    get<LegalMovesResponse>("/legal_moves", { game_id: gameId }),
  move: (gameId: string, moveId: number) =>
    post<MoveResponse>("/move", { game_id: gameId, move_id: moveId }),
  agentMove: (gameId: string) => post<AgentMoveResponse>("/agent_move", { game_id: gameId }),
  checkpoints: () => get<CheckpointsResponse>("/checkpoints"),
};
