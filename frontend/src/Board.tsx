// SVG board renderer. Pure presentation: it draws a StateView in absolute
// coordinates and reports clicks (point index, BAR, or OFF) to the parent, which
// owns all move logic. Highlighting is driven by the source set the parent computes
// from the backend's legal moves. The board also plays a declarative checker-glide
// animation (the `animation` prop) and renders the dice in the mover's colour.

import { useEffect, useLayoutEffect, useRef, useState } from "react";

import { applySubmove } from "./game";
import { BAR, OFF, type Color, type StateView, type SubmoveView } from "./types";

const VB_W = 1000;
const VB_H = 600;
const LEFT = 30;
const RIGHT = 905;
const BAR_W = 46;
const QUAD = (RIGHT - LEFT - BAR_W) / 2; // width spanned by six points
const PW = QUAD / 6; // per-point column width
const TOP_Y = 22;
const BOT_Y = VB_H - 22;
const TRI_H = 248;
const R = PW / 2 - 4; // checker radius
const BAR_X = LEFT + QUAD + BAR_W / 2;
const OFF_X = (RIGHT + VB_W) / 2;
const MAX_STACK = 5; // beyond this, draw a numeric badge instead of more checkers
const BAR_HALF = (BOT_Y - TOP_Y) / 2; // vertical space each colour gets on the bar

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

// Vertical gap between stacked checkers, shrunk so `count` checkers fit inside the
// available half-height. Fixes the overlap when two facing points each hold five.
function stackStep(count: number, avail: number): number {
  if (count <= 1) return 2 * R + 1;
  return Math.min(2 * R + 1, (avail - 2 * R) / (count - 1));
}

// Absolute index 0..11 sit on the bottom row (right-to-left), 12..23 on the top row.
function colCenterX(col: number): number {
  const gap = col >= 6 ? BAR_W : 0;
  return LEFT + col * PW + gap + PW / 2;
}

function slotX(index: number): number {
  return index <= 11 ? colCenterX(11 - index) : colCenterX(index - 12);
}

function isTop(index: number): boolean {
  return index > 11;
}

function trianglePath(index: number): string {
  const x = slotX(index);
  const half = PW / 2 - 1;
  if (isTop(index)) {
    return `M ${x - half} ${TOP_Y} L ${x + half} ${TOP_Y} L ${x} ${TOP_Y + TRI_H} Z`;
  }
  return `M ${x - half} ${BOT_Y} L ${x + half} ${BOT_Y} L ${x} ${BOT_Y - TRI_H} Z`;
}

function checkerCY(index: number, i: number, count: number): number {
  const step = stackStep(count, TRI_H);
  return isTop(index) ? TOP_Y + R + 2 + i * step : BOT_Y - R - 2 - i * step;
}

function barCY(color: Color, i: number, count: number): number {
  const step = stackStep(count, BAR_HALF);
  // Black stacks down from the top half, white up from the bottom half.
  return color === "black" ? TOP_Y + R + 2 + i * step : BOT_Y - R - 2 - i * step;
}

const OFF_Y: Record<Color, number> = { black: TOP_Y + 80, white: BOT_Y - 80 };

// Pixel centre of the checker being lifted from `sm.src` in `board`.
function sourceXY(board: StateView, sm: SubmoveView, mover: Color): { x: number; y: number } {
  if (sm.src === BAR) {
    const count = board.bar[mover];
    return { x: BAR_X, y: barCY(mover, Math.max(count - 1, 0), count) };
  }
  const count = Math.abs(board.board[sm.src]);
  return { x: slotX(sm.src), y: checkerCY(sm.src, Math.max(count - 1, 0), count) };
}

// Pixel centre the checker lands on, given the board *after* the submove.
function destXY(post: StateView, sm: SubmoveView, mover: Color): { x: number; y: number } {
  if (sm.dst === OFF) return { x: OFF_X, y: OFF_Y[mover] };
  const count = Math.abs(post.board[sm.dst]);
  return { x: slotX(sm.dst), y: checkerCY(sm.dst, Math.max(count - 1, 0), count) };
}

function fill(color: Color): string {
  return color === "white" ? "#f4f1ea" : "#2b2b2b";
}
function strokeOf(color: Color): string {
  return color === "white" ? "#9c9384" : "#000";
}

interface StackProps {
  x: number;
  count: number;
  color: Color;
  cy: (i: number, count: number) => number;
}

function CheckerStack({ x, count, color, cy }: StackProps) {
  const shown = Math.min(count, MAX_STACK);
  const items = [];
  for (let i = 0; i < shown; i++) {
    items.push(
      <circle key={i} cx={x} cy={cy(i, count)} r={R} fill={fill(color)} stroke={strokeOf(color)} strokeWidth={1.5} />,
    );
  }
  if (count > MAX_STACK) {
    items.push(
      <text
        key="badge"
        x={x}
        y={cy(MAX_STACK - 1, count) + 4}
        textAnchor="middle"
        fontSize={R}
        fill={color === "white" ? "#2b2b2b" : "#f4f1ea"}
      >
        {count}
      </text>,
    );
  }
  return <g>{items}</g>;
}

const PIPS: Record<number, [number, number][]> = {
  // [col, row] on a 3x3 grid: 0=low, 1=mid, 2=high.
  1: [[1, 1]],
  2: [[0, 0], [2, 2]],
  3: [[0, 0], [1, 1], [2, 2]],
  4: [[0, 0], [2, 0], [0, 2], [2, 2]],
  5: [[0, 0], [2, 0], [1, 1], [0, 2], [2, 2]],
  6: [[0, 0], [2, 0], [0, 1], [2, 1], [0, 2], [2, 2]],
};

function Die({
  x,
  y,
  size,
  value,
  color,
  spent,
}: {
  x: number;
  y: number;
  size: number;
  value: number;
  color: Color;
  spent: boolean;
}) {
  const face = fill(color);
  const pip = color === "white" ? "#2b2b2b" : "#f4f1ea";
  const m = size * 0.26;
  const xs = [x + m, x + size / 2, x + size - m];
  const ys = [y + m, y + size / 2, y + size - m];
  const r = size * 0.085;
  return (
    <g opacity={spent ? 0.3 : 1}>
      <rect x={x} y={y} width={size} height={size} rx={size * 0.16} fill={face} stroke="#1c1c1c" strokeWidth={2} />
      {(PIPS[value] ?? []).map(([c, rrow], i) => (
        <circle key={i} cx={xs[c]} cy={ys[rrow]} r={r} fill={pip} />
      ))}
    </g>
  );
}

// The "Roll" control, drawn in the centre bar exactly where the dice land. Kept in
// SVG coordinates (not an HTML overlay) because the bar centre is not the viewBox
// centre — the off-tray shifts the play area left — so it stays aligned and scales
// with the board. Shown only on the human's roll turn; mutually exclusive with dice.
function RollButton({ disabled, onRoll, color }: { disabled: boolean; onRoll: () => void; color: Color }) {
  const w = 140;
  const h = 60;
  const x = BAR_X - w / 2;
  const y = VB_H / 2 - h / 2;
  const dieSize = 26;
  const dieX = x + 16;
  const labelX = (dieX + dieSize + (x + w)) / 2; // centred between the die and the pill's right edge
  return (
    <g
      className={disabled ? undefined : "roll-pill"}
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-label="Roll the dice"
      aria-disabled={disabled}
      opacity={disabled ? 0.5 : 1}
      style={{ cursor: disabled ? "default" : "pointer" }}
      onClick={() => !disabled && onRoll()}
      onKeyDown={(e) => {
        if (disabled) return;
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onRoll();
        }
      }}
    >
      <rect x={x} y={y} width={w} height={h} rx={14} fill="#e7d3ab" stroke="#b08a4f" strokeWidth={2} />
      <Die x={dieX} y={VB_H / 2 - dieSize / 2} size={dieSize} value={5} color={color} spent={false} />
      <text x={labelX} y={VB_H / 2} textAnchor="middle" dominantBaseline="central" fontSize={22} fontWeight={700} fill="#1c1c1c">
        Roll
      </text>
    </g>
  );
}

export interface GlideAnim {
  fromBoard: StateView;
  submoves: SubmoveView[];
  mover: Color;
  startDelayMs: number;
  glideMs: number;
  settleMs: number;
  onDone: () => void;
}

export interface BoardProps {
  state: StateView;
  sources: Set<number>;
  glowPoint: number | null;
  dice: [number, number] | null;
  diceColor: Color;
  diceSpent: [boolean, boolean];
  diceRemaining: number | null; // doubles: how many plays remain; else null
  animation: GlideAnim | null;
  onPick: (square: number) => void;
  onFlipDice?: () => void;
  onRoll?: () => void; // present only on the human's (non-manual) roll turn; shows the centre pill
  rollDisabled?: boolean; // dim the pill and ignore input while busy or animating
}

export default function Board({
  state,
  sources,
  glowPoint,
  dice,
  diceColor,
  diceSpent,
  diceRemaining,
  animation,
  onPick,
  onFlipDice,
  onRoll,
  rollDisabled,
}: BoardProps) {
  const [animBoard, setAnimBoard] = useState<StateView | null>(null);
  const [glide, setGlide] = useState<{ from: { x: number; y: number }; to: { x: number; y: number }; color: Color; ms: number } | null>(null);
  const [animGlow, setAnimGlow] = useState<number | null>(null);

  // Play the declarative animation: reveal pause, then glide each submove, commit it
  // to the shown board, briefly glow the landing point, repeat; onDone at the end.
  useEffect(() => {
    if (!animation) {
      setAnimBoard(null);
      setGlide(null);
      setAnimGlow(null);
      return;
    }
    let cancelled = false;
    const run = async () => {
      let board = animation.fromBoard;
      setAnimBoard(board);
      setGlide(null);
      setAnimGlow(null);
      if (animation.startDelayMs) await sleep(animation.startDelayMs);
      for (const sm of animation.submoves) {
        if (cancelled) return;
        const from = sourceXY(board, sm, animation.mover);
        const post = applySubmove(board, sm, animation.mover);
        const to = destXY(post, sm, animation.mover);
        setGlide({ from, to, color: animation.mover, ms: animation.glideMs });
        await sleep(animation.glideMs);
        if (cancelled) return;
        board = post;
        setAnimBoard(board);
        setGlide(null);
        if (sm.dst >= 0) setAnimGlow(sm.dst);
        await sleep(animation.settleMs);
        if (cancelled) return;
        setAnimGlow(null);
      }
      if (!cancelled) animation.onDone();
    };
    void run();
    return () => {
      cancelled = true;
    };
  }, [animation]);

  const tokenRef = useRef<SVGGElement>(null);
  useLayoutEffect(() => {
    if (!glide || !tokenRef.current) return;
    const anim = tokenRef.current.animate(
      [
        { transform: `translate(${glide.from.x}px, ${glide.from.y}px)` },
        { transform: `translate(${glide.to.x}px, ${glide.to.y}px)` },
      ],
      { duration: glide.ms, easing: "ease-in-out", fill: "forwards" },
    );
    return () => anim.cancel();
  }, [glide]);

  const shown = animation ? (animBoard ?? animation.fromBoard) : state;
  const glow = animation ? animGlow : glowPoint;
  const clickable = !animation;

  const points = [];
  for (let index = 0; index < 24; index++) {
    const x = slotX(index);
    const top = isTop(index);
    const checkers = shown.board[index];
    const color: Color = checkers > 0 ? "white" : "black";
    const hitY = top ? TOP_Y : BOT_Y - TRI_H;
    const isSrc = sources.has(index);
    points.push(
      <g
        key={index}
        className="point"
        onClick={() => clickable && onPick(index)}
        style={{ cursor: clickable && isSrc ? "pointer" : "default" }}
      >
        <path d={trianglePath(index)} fill={index % 2 === 0 ? "#c9a66b" : "#8a5a2b"} opacity={0.85} />
        <rect x={x - PW / 2} y={hitY} width={PW} height={TRI_H} fill={isSrc ? "rgba(241,196,15,0.30)" : "transparent"} />
        {checkers !== 0 && (
          <CheckerStack x={x} count={Math.abs(checkers)} color={color} cy={(i, c) => checkerCY(index, i, c)} />
        )}
      </g>,
    );
  }

  // Gold ring around the just-moved checker (the persistent or transient glow).
  let glowRing = null;
  if (glow !== null && glow >= 0 && shown.board[glow] !== 0) {
    const c = Math.abs(shown.board[glow]);
    glowRing = (
      <circle
        cx={slotX(glow)}
        cy={checkerCY(glow, c - 1, c)}
        r={R + 3}
        fill="none"
        stroke="#f1c40f"
        strokeWidth={3}
        pointerEvents="none"
      />
    );
  }

  const dieSize = 48;
  const dieGap = 14;
  const diceX0 = BAR_X - dieSize - dieGap / 2;

  return (
    <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="board" role="img" aria-label="backgammon board">
      <rect x={0} y={0} width={VB_W} height={VB_H} fill="#3a2a18" rx={10} />
      <rect x={LEFT - 8} y={TOP_Y - 8} width={RIGHT - LEFT + 16} height={BOT_Y - TOP_Y + 16} fill="#d8b377" />
      {/* bar gutter */}
      <rect x={BAR_X - BAR_W / 2} y={TOP_Y - 8} width={BAR_W} height={BOT_Y - TOP_Y + 16} fill="#5a3d22" />
      {points}

      {/* Bar: black on top half, white on bottom half. */}
      <g className="bar" onClick={() => clickable && onPick(BAR)} style={{ cursor: clickable && sources.has(BAR) ? "pointer" : "default" }}>
        <rect
          x={BAR_X - BAR_W / 2}
          y={TOP_Y}
          width={BAR_W}
          height={BOT_Y - TOP_Y}
          fill={sources.has(BAR) ? "rgba(241,196,15,0.30)" : "transparent"}
        />
        <CheckerStack x={BAR_X} count={shown.bar.black} color="black" cy={(i, c) => barCY("black", i, c)} />
        <CheckerStack x={BAR_X} count={shown.bar.white} color="white" cy={(i, c) => barCY("white", i, c)} />
      </g>

      {/* Off tray on the right: black off on top, white off on bottom. */}
      <g className="off" onClick={() => clickable && onPick(OFF)} style={{ cursor: "default" }}>
        <rect
          x={OFF_X - 38}
          y={TOP_Y}
          width={76}
          height={BOT_Y - TOP_Y}
          fill={sources.has(OFF) ? "rgba(241,196,15,0.30)" : "transparent"}
          stroke="#d8b377"
          strokeWidth={2}
        />
        <text x={OFF_X} y={TOP_Y + 24} textAnchor="middle" fontSize={20} fill="#d8b377">
          off
        </text>
        <text x={OFF_X} y={TOP_Y + 56} textAnchor="middle" fontSize={26} fill="#f4f1ea">
          ⚫ {shown.off.black}
        </text>
        <text x={OFF_X} y={BOT_Y - 30} textAnchor="middle" fontSize={26} fill="#f4f1ea">
          ⚪ {shown.off.white}
        </text>
      </g>

      {glowRing}

      {/* Roll control in the centre, where the dice will land. Only on the human's
          roll turn, and never alongside dice (cleared to null before a roll). */}
      {onRoll && !dice && <RollButton disabled={!!rollDisabled} onRoll={onRoll} color={diceColor} />}

      {/* Dice in the centre, coloured for the mover; click to flip the order. */}
      {dice && (
        <g onClick={() => onFlipDice?.()} style={{ cursor: onFlipDice ? "pointer" : "default" }}>
          {onFlipDice && (
            <title>Click to swap the dice order</title>
          )}
          <Die x={diceX0} y={VB_H / 2 - dieSize / 2} size={dieSize} value={dice[0]} color={diceColor} spent={diceSpent[0]} />
          <Die x={diceX0 + dieSize + dieGap} y={VB_H / 2 - dieSize / 2} size={dieSize} value={dice[1]} color={diceColor} spent={diceSpent[1]} />
          {diceRemaining !== null && (
            <text x={BAR_X} y={VB_H / 2 + dieSize / 2 + 22} textAnchor="middle" fontSize={20} fill="#f4f1ea">
              ×{diceRemaining} left
            </text>
          )}
        </g>
      )}

      {/* The gliding checker (rendered last so it floats above the board). */}
      {glide && (
        <g ref={tokenRef} style={{ transform: `translate(${glide.from.x}px, ${glide.from.y}px)` }} pointerEvents="none">
          <circle cx={0} cy={0} r={R} fill={fill(glide.color)} stroke={strokeOf(glide.color)} strokeWidth={1.5} />
        </g>
      )}
    </svg>
  );
}
