"""
Unit tests for Task 3 AI Planning & State Space Search (modules/planning.py).
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules import planning


def test_initial_and_goal_states():
    init = planning.DEFAULT_INITIAL_STATE
    goal = planning.DEFAULT_GOAL_STATE
    assert len(init) == 8
    assert len(goal) == 8
    assert not planning.is_goal(init)
    assert planning.is_goal(goal)


def test_heuristic_admissibility():
    init = planning.DEFAULT_INITIAL_STATE
    goal = planning.DEFAULT_GOAL_STATE
    assert planning.heuristic(goal) == 0
    assert planning.heuristic(init) == 14  # Sum of mandatory repair costs


def test_valid_action_prerequisites():
    init = planning.DEFAULT_INITIAL_STATE
    # From initial state, only 'Inspect Bearing' is legal
    assert planning.is_valid_action(init, "Inspect Bearing") is True
    assert planning.is_valid_action(init, "Replace Bearing") is False
    assert planning.is_valid_action(init, "Test Pump") is False


def test_apply_action_transitions():
    init = planning.DEFAULT_INITIAL_STATE
    next_s = planning.apply_action(init, "Inspect Bearing")
    assert next_s[0] is True
    assert next_s[1:] == init[1:]


def test_bfs_search():
    res = planning.bfs()
    assert res["found"] is True
    assert res["path_cost"] == 14
    assert len(res["actions"]) == 7
    assert res["nodes_expanded"] > 0


def test_dfs_search():
    res = planning.dfs()
    assert res["found"] is True
    assert len(res["actions"]) >= 7
    assert res["nodes_expanded"] > 0


def test_astar_search():
    res = planning.astar()
    assert res["found"] is True
    assert res["path_cost"] == 14
    assert len(res["actions"]) == 7
    # A* expands fewer or equal nodes compared to uninformed BFS
    bfs_res = planning.bfs()
    assert res["nodes_expanded"] <= bfs_res["nodes_expanded"]


def test_path_validation():
    res = planning.astar()
    valid, msg = planning.validate_path(planning.DEFAULT_INITIAL_STATE, res["actions"])
    assert valid is True
    assert "Valid path" in msg

    # Invalid sequence: trying to replace bearing before inspection
    invalid_seq = ["Replace Bearing"]
    valid, msg = planning.validate_path(planning.DEFAULT_INITIAL_STATE, invalid_seq)
    assert valid is False
    assert "Invalid action" in msg


def test_run_all_algorithms():
    all_res = planning.run_all_algorithms()
    assert set(all_res.keys()) == {"BFS", "DFS", "A*"}
    for name, res in all_res.items():
        assert res["found"] is True
