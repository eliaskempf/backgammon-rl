// SVG board renderer. Pure presentation: it draws a StateView in absolute
// coordinates and reports clicks (point index, BAR, or OFF) to the parent, which
// owns all move logic. Highlighting is driven by the source/destination sets the
// parent computes from the backend's legal moves.

import { BAR, OFF, type StateView } from "./types";

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

function checkerCY(index: number, i: number): number {
  const step = 2 * R + 1;
  return isTop(index) ? TOP_Y + R + 2 + i * step : BOT_Y - R - 2 - i * step;
}

interface StackProps {
  x: number;
  count: number;
  color: "white" | "black";
  cy: (i: number) => number;
}

function CheckerStack({ x, count, color, cy }: StackProps) {
  const shown = Math.min(count, MAX_STACK);
  const fill = color === "white" ? "#f4f1ea" : "#2b2b2b";
  const stroke = color === "white" ? "#9c9384" : "#000";
  const items = [];
  for (let i = 0; i < shown; i++) {
    items.push(
      <circle key={i} cx={x} cy={cy(i)} r={R} fill={fill} stroke={stroke} strokeWidth={1.5} />,
    );
  }
  if (count > MAX_STACK) {
    items.push(
      <text
        key="badge"
        x={x}
        y={cy(MAX_STACK - 1) + 4}
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

export interface BoardProps {
  state: StateView;
  sources: Set<number>;
  destinations: Set<number>;
  selected: number | null;
  dice: [number, number] | null;
  onPick: (square: number) => void;
}

function highlightFill(
  square: number,
  sources: Set<number>,
  destinations: Set<number>,
  selected: number | null,
): string {
  if (square === selected) return "rgba(52,152,219,0.55)";
  if (destinations.has(square)) return "rgba(46,204,113,0.45)";
  if (sources.has(square)) return "rgba(241,196,15,0.35)";
  return "transparent";
}

export default function Board({
  state,
  sources,
  destinations,
  selected,
  dice,
  onPick,
}: BoardProps) {
  const points = [];
  for (let index = 0; index < 24; index++) {
    const x = slotX(index);
    const top = isTop(index);
    const checkers = state.board[index];
    const color = checkers > 0 ? "white" : "black";
    const hitY = top ? TOP_Y : BOT_Y - TRI_H;
    points.push(
      <g key={index} className="point" onClick={() => onPick(index)} style={{ cursor: "pointer" }}>
        <path d={trianglePath(index)} fill={index % 2 === 0 ? "#c9a66b" : "#8a5a2b"} opacity={0.85} />
        <rect
          x={x - PW / 2}
          y={hitY}
          width={PW}
          height={TRI_H}
          fill={highlightFill(index, sources, destinations, selected)}
        />
        {checkers !== 0 && (
          <CheckerStack
            x={x}
            count={Math.abs(checkers)}
            color={color}
            cy={(i) => checkerCY(index, i)}
          />
        )}
      </g>,
    );
  }

  return (
    <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="board" role="img" aria-label="backgammon board">
      <rect x={0} y={0} width={VB_W} height={VB_H} fill="#3a2a18" rx={10} />
      <rect x={LEFT - 8} y={TOP_Y - 8} width={RIGHT - LEFT + 16} height={BOT_Y - TOP_Y + 16} fill="#d8b377" />
      {/* bar gutter */}
      <rect x={BAR_X - BAR_W / 2} y={TOP_Y - 8} width={BAR_W} height={BOT_Y - TOP_Y + 16} fill="#5a3d22" />
      {points}

      {/* Bar: black on top half, white on bottom half. */}
      <g className="bar" onClick={() => onPick(BAR)} style={{ cursor: "pointer" }}>
        <rect
          x={BAR_X - BAR_W / 2}
          y={TOP_Y}
          width={BAR_W}
          height={BOT_Y - TOP_Y}
          fill={highlightFill(BAR, sources, destinations, selected)}
        />
        <CheckerStack x={BAR_X} count={state.bar.black} color="black" cy={(i) => TOP_Y + R + 2 + i * (2 * R + 1)} />
        <CheckerStack x={BAR_X} count={state.bar.white} color="white" cy={(i) => BOT_Y - R - 2 - i * (2 * R + 1)} />
      </g>

      {/* Off tray on the right: black off on top, white off on bottom. */}
      <g className="off" onClick={() => onPick(OFF)} style={{ cursor: "pointer" }}>
        <rect
          x={OFF_X - 38}
          y={TOP_Y}
          width={76}
          height={BOT_Y - TOP_Y}
          fill={highlightFill(OFF, sources, destinations, selected)}
          stroke="#d8b377"
          strokeWidth={2}
        />
        <text x={OFF_X} y={TOP_Y + 24} textAnchor="middle" fontSize={20} fill="#d8b377">
          off
        </text>
        <text x={OFF_X} y={TOP_Y + 56} textAnchor="middle" fontSize={26} fill="#2b2b2b">
          ⚫ {state.off.black}
        </text>
        <text x={OFF_X} y={BOT_Y - 30} textAnchor="middle" fontSize={26} fill="#f4f1ea">
          ⚪ {state.off.white}
        </text>
      </g>

      {/* Dice readout in the centre. */}
      {dice && (
        <g>
          {[dice[0], dice[1]].map((d, i) => (
            <g key={i}>
              <rect x={BAR_X - 70 + i * 70} y={VB_H / 2 - 24} width={48} height={48} rx={8} fill="#f4f1ea" stroke="#2b2b2b" strokeWidth={2} />
              <text x={BAR_X - 70 + i * 70 + 24} y={VB_H / 2 + 8} textAnchor="middle" fontSize={28} fill="#2b2b2b">
                {d}
              </text>
            </g>
          ))}
        </g>
      )}
    </svg>
  );
}
