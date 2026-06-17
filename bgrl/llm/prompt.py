"""Prompt templates: compose a board rendering + an output format into chat messages.

A :class:`PromptTemplate` is one swept axis (the wording/framing); it stays orthogonal
to the *other* axes by delegating the board to a :class:`~bgrl.llm.render.BoardRenderer`
and the answer format to an :class:`~bgrl.llm.parse.OutputFormat`. :meth:`build` always
renders from the mover's perspective and presents the legal moves as an enumerated list
so the model answers with an index (never a free-form move) — the index is the action
vocabulary (afterstate-first, CLAUDE.md §4).
"""

from __future__ import annotations

from dataclasses import dataclass

from bgrl.env import Dice, EnvState, Move
from bgrl.llm.client import ChatMessage
from bgrl.llm.parse import OutputFormat
from bgrl.llm.render import BoardRenderer, describe_move


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """A named ``(system, intro)`` framing that builds the chat messages for one turn."""

    name: str
    system: str
    intro: str

    def build(
        self,
        state: EnvState,
        dice: Dice,
        legal: list[tuple[Move, EnvState]],
        renderer: BoardRenderer,
        output_format: OutputFormat,
    ) -> list[ChatMessage]:
        """Assemble ``[system, user]`` messages for the current ``(state, dice, legal)``."""
        mover = state.turn
        board = renderer.render(state, dice, mover)
        moves = "\n".join(
            f"{i}: {describe_move(move, mover)}" for i, (move, _after) in enumerate(legal)
        )
        user = (
            f"{self.intro}\n\n"
            f"You rolled {dice[0]}-{dice[1]}.\n\n"
            f"{board}\n\n"
            f"Candidate moves (choose one by its index):\n{moves}\n\n"
            f"{output_format.instruction(len(legal))}"
        )
        return [ChatMessage("system", self.system), ChatMessage("user", user)]


TERSE = PromptTemplate(
    name="terse",
    system="You are an expert backgammon player. Choose the strongest legal move.",
    intro=(
        "It is your turn. The position is shown from your perspective: your checkers "
        "are X and move from point 24 toward point 1 to bear off; the opponent's are O."
    ),
)

COACH = PromptTemplate(
    name="coach",
    system=(
        "You are a world-class backgammon engine. Pick the move with the highest equity, "
        "weighing making points, hitting blots, escaping your back checkers, building your "
        "home board, and avoiding being hit."
    ),
    intro=(
        "It is your turn. The position is shown from your perspective: your checkers are X "
        "moving from point 24 toward point 1 (your home is points 1-6); the opponent's are O. "
        "Consider the resulting position's safety and structure before deciding."
    ),
)

ALL_TEMPLATES: dict[str, PromptTemplate] = {t.name: t for t in (TERSE, COACH)}
"""Registry of prompt templates by ``name``, for the CLI/sweep to select by string."""
