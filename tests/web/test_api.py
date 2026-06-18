"""HTTP-level tests for the play server, driven through FastAPI's TestClient."""

import pytest
from fastapi.testclient import TestClient

from bgrl.nets.value_net import MLPValueNet
from bgrl.serialization import save_checkpoint
from bgrl.web import create_app


@pytest.fixture
def client(tmp_path):
    # Empty checkpoints dir by default -> only the "random" opponent is available.
    return TestClient(create_app(checkpoints_dir=tmp_path))


def _run_game(client, new_game_resp, limit=5000):
    """Drive a full game to termination; the human always plays the first legal move."""
    gid = new_game_resp["game_id"]
    human = new_game_resp["human_color"]
    to_act = new_game_resp["to_act"]
    terminal, outcome = False, None
    for _ in range(limit):
        if terminal:
            break
        if to_act == human:
            data = client.post("/roll", json={"game_id": gid}).json()
            to_act, terminal, outcome = data["to_act"], data["terminal"], data["outcome"]
            if data["auto_pass"] or terminal:
                continue
            moves = client.get("/legal_moves", params={"game_id": gid}).json()["moves"]
            data = client.post("/move", json={"game_id": gid, "move_id": moves[0]["id"]}).json()
        else:
            data = client.post("/agent_move", json={"game_id": gid}).json()
        to_act, terminal, outcome = data["to_act"], data["terminal"], data["outcome"]
    return terminal, outcome


def test_full_game_reaches_a_terminal_outcome(client):
    resp = client.post("/new_game", json={"human_color": "white", "opponent": "random", "seed": 7})
    assert resp.status_code == 200
    ng = resp.json()
    assert ng["to_act"] == "white" and ng["needs_roll"] is True

    terminal, outcome = _run_game(client, ng)
    assert terminal is True
    assert outcome is not None
    assert outcome["winner"] in {"white", "black"}
    assert outcome["kind"] in {"single", "gammon", "backgammon"}


def test_human_can_play_second_seat(client):
    # human_color black -> WHITE (the opponent) moves first; the loop must cope.
    ng = client.post("/new_game", json={"human_color": "black", "opponent": "random"}).json()
    assert ng["to_act"] == "white"  # WHITE always opens
    terminal, outcome = _run_game(client, ng)
    assert terminal and outcome is not None


def test_illegal_move_id_returns_409(client):
    ng = client.post("/new_game", json={"human_color": "white", "opponent": "random"}).json()
    gid = ng["game_id"]
    assert client.post("/roll", json={"game_id": gid}).status_code == 200
    bad = client.post("/move", json={"game_id": gid, "move_id": 9999})
    assert bad.status_code == 409


def test_move_before_roll_and_double_roll_conflict(client):
    ng = client.post("/new_game", json={"human_color": "white", "opponent": "random"}).json()
    gid = ng["game_id"]
    # Moving before rolling is a conflict.
    assert client.post("/move", json={"game_id": gid, "move_id": 0}).status_code == 409
    assert client.post("/roll", json={"game_id": gid}).status_code == 200
    # Rolling again before playing is a conflict.
    assert client.post("/roll", json={"game_id": gid}).status_code == 409


def test_agent_move_rejected_on_humans_turn(client):
    ng = client.post("/new_game", json={"human_color": "white", "opponent": "random"}).json()
    # It's WHITE's (the human's) turn, so asking the agent to move is a conflict.
    assert client.post("/agent_move", json={"game_id": ng["game_id"]}).status_code == 409


def test_unknown_game_id_404_and_unknown_opponent_400(client):
    assert client.post("/roll", json={"game_id": "nope"}).status_code == 404
    bad = client.post("/new_game", json={"human_color": "white", "opponent": "ghost"})
    assert bad.status_code == 400


def test_sessions_are_isolated(client):
    a = client.post("/new_game", json={"human_color": "white", "opponent": "random"}).json()
    b = client.post("/new_game", json={"human_color": "white", "opponent": "random"}).json()
    assert a["game_id"] != b["game_id"]
    # Rolling in game A must not consume game B's roll.
    client.post("/roll", json={"game_id": a["game_id"]})
    legal_b = client.get("/legal_moves", params={"game_id": b["game_id"]}).json()
    assert legal_b["dice"] is None and legal_b["moves"] == []


def test_checkpoint_opponent_is_listed_and_playable(tmp_path):
    save_checkpoint(MLPValueNet(hidden=8), tmp_path / "tiny.pt", trained_with="random")
    client = TestClient(create_app(checkpoints_dir=tmp_path))

    listed = client.get("/checkpoints").json()["checkpoints"]
    assert any(c["name"] == "tiny" for c in listed)

    ng = client.post(
        "/new_game", json={"human_color": "white", "opponent": "tiny", "seed": 1}
    ).json()
    assert ng["opponent"] == "tiny"
    # Play one full ply pair to confirm the value-net opponent actually moves.
    client.post("/roll", json={"game_id": ng["game_id"]})
    moves = client.get("/legal_moves", params={"game_id": ng["game_id"]}).json()["moves"]
    client.post("/move", json={"game_id": ng["game_id"], "move_id": moves[0]["id"]})
    agent = client.post("/agent_move", json={"game_id": ng["game_id"]}).json()
    assert agent["move"] is not None and agent["dice"][0] in range(1, 7)


def test_expectimax_opponent_plays_and_reports_win_prob(tmp_path):
    save_checkpoint(MLPValueNet(hidden=8), tmp_path / "tiny.pt", trained_with="random")
    client = TestClient(create_app(checkpoints_dir=tmp_path))
    ng = client.post(
        "/new_game",
        json={
            "opponent": "tiny",
            "manual_dice": True,
            "expectimax_plies": 2,  # wraps the net in 2-ply WP2 expectimax
            "expectimax_top_k": 2,  # tight pruning keeps the test fast
        },
    ).json()
    gid = ng["game_id"]
    client.post("/roll", json={"game_id": gid, "dice": [3, 1]})
    moves = client.get("/legal_moves", params={"game_id": gid}).json()["moves"]
    client.post("/move", json={"game_id": gid, "move_id": moves[0]["id"]})
    agent = client.post("/agent_move", json={"game_id": gid, "dice": [5, 2]}).json()
    assert agent["move"] is not None  # the searching opponent actually moved
    assert agent["win_prob"] is not None and 0.0 <= agent["win_prob"] <= 1.0


def test_new_game_rejects_out_of_range_plies(client):
    r = client.post("/new_game", json={"opponent": "random", "expectimax_plies": 3})
    assert r.status_code == 422


def test_empty_checkpoints_dir_lists_nothing(client):
    assert client.get("/checkpoints").json()["checkpoints"] == []


# --- sub-move die labels, value estimate, multi-turn undo ---------------------


def test_legal_moves_carry_die_labels(client):
    ng = client.post(
        "/new_game", json={"human_color": "white", "opponent": "random", "manual_dice": True}
    ).json()
    gid = ng["game_id"]
    client.post("/roll", json={"game_id": gid, "dice": [3, 1]})
    moves = client.get("/legal_moves", params={"game_id": gid}).json()["moves"]
    assert moves
    for m in moves:
        assert [sm["die"] for sm in m["submoves"]]  # every submove labelled
        assert all(sm["die"] in (1, 3) for sm in m["submoves"])

    # Doubles: every submove is labelled with the doubled value.
    ng2 = client.post(
        "/new_game", json={"human_color": "white", "opponent": "random", "manual_dice": True}
    ).json()
    gid2 = ng2["game_id"]
    client.post("/roll", json={"game_id": gid2, "dice": [2, 2]})
    moves2 = client.get("/legal_moves", params={"game_id": gid2}).json()["moves"]
    assert moves2
    assert all(sm["die"] == 2 for m in moves2 for sm in m["submoves"])


def test_win_prob_present_for_value_net_opponent(tmp_path):
    save_checkpoint(MLPValueNet(hidden=8), tmp_path / "tiny.pt", trained_with="random")
    client = TestClient(create_app(checkpoints_dir=tmp_path))
    ng = client.post(
        "/new_game", json={"human_color": "white", "opponent": "tiny", "manual_dice": True}
    ).json()
    gid = ng["game_id"]
    client.post("/roll", json={"game_id": gid, "dice": [3, 1]})
    moves = client.get("/legal_moves", params={"game_id": gid}).json()["moves"]
    moved = client.post("/move", json={"game_id": gid, "move_id": moves[0]["id"]}).json()
    assert moved["win_prob"] is not None and 0.0 <= moved["win_prob"] <= 1.0

    agent = client.post("/agent_move", json={"game_id": gid, "dice": [5, 2]}).json()
    assert agent["win_prob"] is not None and 0.0 <= agent["win_prob"] <= 1.0


def test_win_prob_absent_for_random_opponent(client):
    ng = client.post(
        "/new_game", json={"human_color": "white", "opponent": "random", "manual_dice": True}
    ).json()
    gid = ng["game_id"]
    client.post("/roll", json={"game_id": gid, "dice": [3, 1]})
    moves = client.get("/legal_moves", params={"game_id": gid}).json()["moves"]
    moved = client.post("/move", json={"game_id": gid, "move_id": moves[0]["id"]}).json()
    assert moved["win_prob"] is None


def test_undo_reverts_a_full_turn_and_repeats(client):
    ng = client.post(
        "/new_game", json={"human_color": "white", "opponent": "random", "manual_dice": True}
    ).json()
    gid = ng["game_id"]
    assert ng["can_undo"] is False

    rolled = client.post("/roll", json={"game_id": gid, "dice": [3, 1]}).json()
    assert rolled["can_undo"] is False
    moves = client.get("/legal_moves", params={"game_id": gid}).json()["moves"]
    moved = client.post("/move", json={"game_id": gid, "move_id": moves[0]["id"]}).json()
    assert moved["can_undo"] is True
    agent = client.post("/agent_move", json={"game_id": gid, "dice": [5, 2]}).json()
    assert agent["can_undo"] is True and agent["to_act"] == "white"

    undone = client.post("/undo", json={"game_id": gid}).json()
    assert undone["to_act"] == "white"
    assert undone["dice"] == [3, 1]  # same roll restored
    assert undone["moves"]  # legal moves re-enumerated, with die labels
    assert all(sm["die"] in (1, 3) for m in undone["moves"] for sm in m["submoves"])
    assert undone["can_undo"] is False

    # Nothing left to undo.
    assert client.post("/undo", json={"game_id": gid}).status_code == 409


def test_undo_unknown_game_is_404(client):
    assert client.post("/undo", json={"game_id": "nope"}).status_code == 404


def test_export_mat_returns_a_gnubg_importable_match(client):
    ng = client.post(
        "/new_game", json={"human_color": "white", "opponent": "random", "seed": 7}
    ).json()
    terminal, _ = _run_game(client, ng)
    assert terminal

    resp = client.post("/export_mat", json={"game_id": ng["game_id"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["filename"].endswith(".mat")
    mat = body["mat"]
    assert mat.startswith(" 0 point match")
    assert " Game 1" in mat
    assert "Wins" in mat  # a finished game records its result
    # human sat WHITE (.mat player 1); the random opponent is named on the other side.
    assert "human : 0" in mat and "random : 0" in mat


def test_export_mat_unknown_game_returns_404(client):
    assert client.post("/export_mat", json={"game_id": "nope"}).status_code == 404


# --- manual-dice mode ---------------------------------------------------------


def test_default_game_is_not_manual(client):
    ng = client.post("/new_game", json={"human_color": "white", "opponent": "random"}).json()
    assert ng["manual_dice"] is False


def test_manual_game_uses_supplied_dice_for_both_seats(client):
    ng = client.post(
        "/new_game", json={"human_color": "white", "opponent": "random", "manual_dice": True}
    ).json()
    assert ng["manual_dice"] is True
    gid = ng["game_id"]

    # The human supplies their own roll; the server returns exactly those dice.
    rolled = client.post("/roll", json={"game_id": gid, "dice": [3, 1]}).json()
    assert rolled["dice"] == [3, 1]
    assert client.get("/legal_moves", params={"game_id": gid}).json()["dice"] == [3, 1]

    moves = client.get("/legal_moves", params={"game_id": gid}).json()["moves"]
    client.post("/move", json={"game_id": gid, "move_id": moves[0]["id"]})

    # The human also supplies the agent's roll — the agent never touches an RNG.
    agent = client.post("/agent_move", json={"game_id": gid, "dice": [5, 2]}).json()
    assert agent["dice"] == [5, 2]


def test_manual_roll_without_dice_is_422(client):
    ng = client.post(
        "/new_game", json={"human_color": "white", "opponent": "random", "manual_dice": True}
    ).json()
    assert client.post("/roll", json={"game_id": ng["game_id"]}).status_code == 422


def test_manual_agent_move_without_dice_is_422(client):
    # human black -> WHITE (the agent) is to move first, so /agent_move needs dice.
    ng = client.post(
        "/new_game", json={"human_color": "black", "opponent": "random", "manual_dice": True}
    ).json()
    assert client.post("/agent_move", json={"game_id": ng["game_id"]}).status_code == 422


def test_manual_dice_out_of_range_is_422(client):
    ng = client.post(
        "/new_game", json={"human_color": "white", "opponent": "random", "manual_dice": True}
    ).json()
    bad = client.post("/roll", json={"game_id": ng["game_id"], "dice": [0, 7]})
    assert bad.status_code == 422


def test_dice_sent_to_auto_game_is_409(client):
    ng = client.post("/new_game", json={"human_color": "white", "opponent": "random"}).json()
    bad = client.post("/roll", json={"game_id": ng["game_id"], "dice": [3, 1]})
    assert bad.status_code == 409


def test_manual_game_runs_to_terminal(client):
    """Drive a full manual game (fixed dice for both seats) to a terminal outcome."""
    ng = client.post(
        "/new_game", json={"human_color": "white", "opponent": "random", "manual_dice": True}
    ).json()
    gid, human, to_act = ng["game_id"], ng["human_color"], ng["to_act"]
    # A doubles-heavy stream keeps games short; any in-range dice are accepted.
    stream = [[6, 6], [5, 5], [4, 4], [3, 3], [2, 2], [1, 1], [6, 5], [4, 3], [2, 1], [6, 4]]
    terminal = False
    for i in range(5000):
        if terminal:
            break
        dice = stream[i % len(stream)]
        if to_act == human:
            data = client.post("/roll", json={"game_id": gid, "dice": dice}).json()
            to_act, terminal = data["to_act"], data["terminal"]
            if data["auto_pass"] or terminal:
                continue
            moves = client.get("/legal_moves", params={"game_id": gid}).json()["moves"]
            data = client.post("/move", json={"game_id": gid, "move_id": moves[0]["id"]}).json()
        else:
            data = client.post("/agent_move", json={"game_id": gid, "dice": dice}).json()
        to_act, terminal = data["to_act"], data["terminal"]
    assert terminal
