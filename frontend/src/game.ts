// Client-side board mutation, mirroring the env's apply_submove (bgrl/env/apply.py):
// used for the optimistic move preview and the move animation. The server stays the
// sole authority on legality; these helpers only project a known-legal submove onto a
// StateView so the UI can show it immediately.

import { BAR, OFF, type Color, type StateView, type SubmoveView } from "./types";

function clone(state: StateView): StateView {
  return {
    board: [...state.board],
    bar: { ...state.bar },
    off: { ...state.off },
    turn: state.turn,
    cube: { ...state.cube },
  };
}

// Apply one (assumed legal) submove for `mover`. Handles bar entry (src === BAR),
// bearing off (dst === OFF) and hitting a lone opposing blot (sent to its bar). Does
// not flip whose turn it is — that's a server concern.
export function applySubmove(state: StateView, sm: SubmoveView, mover: Color): StateView {
  const next = clone(state);
  const opp: Color = mover === "white" ? "black" : "white";
  const sign = mover === "white" ? 1 : -1;

  if (sm.src === BAR) next.bar[mover] -= 1;
  else next.board[sm.src] -= sign;

  if (sm.dst === OFF) {
    next.off[mover] += 1;
  } else if (next.board[sm.dst] === -sign) {
    // exactly one opponent checker -> hit it
    next.board[sm.dst] = sign;
    next.bar[opp] += 1;
  } else {
    next.board[sm.dst] += sign;
  }
  return next;
}

export function applySubmoves(state: StateView, submoves: SubmoveView[], mover: Color): StateView {
  return submoves.reduce((acc, sm) => applySubmove(acc, sm, mover), state);
}
