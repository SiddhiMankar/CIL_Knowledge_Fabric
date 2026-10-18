"""
Task 3 Module: State Space Search & AI Maintenance Planning
Original source: CIL3.ipynb

Implements:
- Problem domain definition for industrial pump maintenance (Pump P-101, Bearing Wear).
- State predicates and legal transition operators.
- Breadth-First Search (BFS).
- Depth-First Search (DFS).
- A* Search with admissible heuristic.
- Path validation and benchmark comparison suite.
"""

from collections import deque
import heapq
import time
import math
from typing import Dict, List, Tuple, Any, Optional

# State element indices and names
STATE_VARIABLES = [
    "bearing_inspected",       # idx 0
    "bearing_worn_confirmed",  # idx 1
    "bearing_replaced",        # idx 2
    "lubrication_checked",     # idx 3
    "vibration_tested",        # idx 4
    "temperature_tested",      # idx 5
    "pump_tested",             # idx 6
    "pump_operational"         # idx 7
]

# Standard Initial and Goal States for Pump P-101
DEFAULT_INITIAL_STATE: Tuple[bool, ...] = (
    False,  # bearing_inspected
    False,  # bearing_worn_confirmed
    False,  # bearing_replaced
    False,  # lubrication_checked
    False,  # vibration_tested
    False,  # temperature_tested
    False,  # pump_tested
    False   # pump_operational
)

DEFAULT_GOAL_STATE: Tuple[bool, ...] = (
    True,   # bearing_inspected
    True,   # bearing_worn_confirmed
    True,   # bearing_replaced
    False,  # lubrication_checked - not strictly required
    True,   # vibration_tested
    True,   # temperature_tested
    True,   # pump_tested
    True    # pump_operational
)

# Available Actions and their costs
ACTIONS: Dict[str, Dict[str, int]] = {
    "Inspect Bearing": {"cost": 1},
    "Confirm Bearing Wear": {"cost": 1},
    "Check Lubrication": {"cost": 1},
    "Lubricate Bearing": {"cost": 2},
    "Replace Bearing": {"cost": 5},
    "Run Vibration Test": {"cost": 2},
    "Run Temperature Test": {"cost": 2},
    "Test Pump": {"cost": 2},
    "Restore Pump Operation": {"cost": 1}
}

# Pre-defined operational scenarios
SCENARIOS = {
    "Bearing Failure - P-101": {
        "equipment": "P-101",
        "fault": "Bearing Wear",
        "initial_state": DEFAULT_INITIAL_STATE,
        "goal_state": DEFAULT_GOAL_STATE,
        "description": "Standard industrial pump experiencing abnormal vibration and suspected bearing wear."
    }
}


def is_valid_action(state: Tuple[bool, ...], action: str) -> bool:
    """
    Determines whether an action can legally be performed in the given state.
    Preserves exact prerequisite rules from CIL3.ipynb.
    """
    bearing_inspected = state[0]
    bearing_worn_confirmed = state[1]
    bearing_replaced = state[2]
    lubrication_checked = state[3]
    vibration_tested = state[4]
    temperature_tested = state[5]
    pump_tested = state[6]
    pump_operational = state[7]

    if action == "Inspect Bearing":
        return not bearing_inspected

    elif action == "Confirm Bearing Wear":
        return bearing_inspected and not bearing_worn_confirmed

    elif action == "Check Lubrication":
        return bearing_inspected and not lubrication_checked

    elif action == "Lubricate Bearing":
        return lubrication_checked and not bearing_replaced

    elif action == "Replace Bearing":
        return bearing_worn_confirmed and not bearing_replaced

    elif action == "Run Vibration Test":
        return bearing_replaced and not vibration_tested

    elif action == "Run Temperature Test":
        return bearing_replaced and not temperature_tested

    elif action == "Test Pump":
        return (
            bearing_replaced
            and vibration_tested
            and temperature_tested
            and not pump_tested
        )

    elif action == "Restore Pump Operation":
        return pump_tested and not pump_operational

    return False


def apply_action(state: Tuple[bool, ...], action: str) -> Tuple[bool, ...]:
    """
    Applies a valid action to a state and returns the resulting new state tuple.
    """
    new_state = list(state)

    if action == "Inspect Bearing":
        new_state[0] = True
    elif action == "Confirm Bearing Wear":
        new_state[1] = True
    elif action == "Check Lubrication":
        new_state[3] = True
    elif action == "Lubricate Bearing":
        pass  # Lubrication does not directly affect the main repair predicates
    elif action == "Replace Bearing":
        new_state[2] = True
    elif action == "Run Vibration Test":
        new_state[4] = True
    elif action == "Run Temperature Test":
        new_state[5] = True
    elif action == "Test Pump":
        new_state[6] = True
    elif action == "Restore Pump Operation":
        new_state[7] = True

    return tuple(new_state)


def get_neighbors(state: Tuple[bool, ...]) -> List[Tuple[Tuple[bool, ...], str, int]]:
    """
    Returns all legal successors from the current state.
    Returns: List of (new_state, action_name, action_cost)
    """
    neighbors = []
    for action, data in ACTIONS.items():
        if is_valid_action(state, action):
            new_state = apply_action(state, action)
            neighbors.append((new_state, action, data["cost"]))
    return neighbors


def is_goal(state: Tuple[bool, ...], goal_state: Optional[Tuple[bool, ...]] = None) -> bool:
    """
    Checks whether the goal conditions are satisfied.
    If goal_state is provided, tests required components.
    By default tests the 7 mandatory predicates from CIL3.ipynb:
    (bearing_inspected, bearing_worn_confirmed, bearing_replaced, vibration_tested,
     temperature_tested, pump_tested, pump_operational).
    """
    if goal_state is not None:
        # Check that all True conditions in goal_state are True in state
        for idx, required in enumerate(goal_state):
            if required and not state[idx]:
                return False
        return True

    return bool(
        state[0] and  # bearing inspected
        state[1] and  # bearing wear confirmed
        state[2] and  # bearing replaced
        state[4] and  # vibration tested
        state[5] and  # temperature tested
        state[6] and  # pump tested
        state[7]      # pump operational
    )


def reconstruct_path(
    parent: Dict[Tuple[bool, ...], Optional[Tuple[bool, ...]]],
    action_taken: Dict[Tuple[bool, ...], str],
    goal: Tuple[bool, ...]
) -> Tuple[List[Tuple[bool, ...]], List[str]]:
    """Reconstructs the state trajectory and sequence of actions from initial state to goal."""
    states: List[Tuple[bool, ...]] = []
    actions: List[str] = []
    current: Optional[Tuple[bool, ...]] = goal

    while current is not None:
        states.append(current)
        if current in action_taken:
            actions.append(action_taken[current])
        current = parent[current]

    states.reverse()
    actions.reverse()
    return states, actions


def bfs(initial_state: Tuple[bool, ...] = DEFAULT_INITIAL_STATE, goal_state: Optional[Tuple[bool, ...]] = None) -> Dict[str, Any]:
    """Breadth-First Search (BFS) for maintenance planning."""
    start_time = time.perf_counter()
    queue = deque([initial_state])
    visited = {initial_state}
    parent: Dict[Tuple[bool, ...], Optional[Tuple[bool, ...]]] = {initial_state: None}
    action_taken: Dict[Tuple[bool, ...], str] = {}
    nodes_expanded = 0

    while queue:
        current = queue.popleft()
        nodes_expanded += 1

        if is_goal(current, goal_state):
            states, actions = reconstruct_path(parent, action_taken, current)
            execution_time = time.perf_counter() - start_time
            return {
                "algorithm": "BFS",
                "found": True,
                "states": states,
                "actions": actions,
                "path_cost": sum(ACTIONS[a]["cost"] for a in actions),
                "nodes_expanded": nodes_expanded,
                "execution_time": execution_time
            }

        for next_state, action, cost in get_neighbors(current):
            if next_state not in visited:
                visited.add(next_state)
                parent[next_state] = current
                action_taken[next_state] = action
                queue.append(next_state)

    execution_time = time.perf_counter() - start_time
    return {
        "algorithm": "BFS",
        "found": False,
        "states": [],
        "actions": [],
        "path_cost": math.inf,
        "nodes_expanded": nodes_expanded,
        "execution_time": execution_time
    }


def dfs(initial_state: Tuple[bool, ...] = DEFAULT_INITIAL_STATE, goal_state: Optional[Tuple[bool, ...]] = None) -> Dict[str, Any]:
    """Depth-First Search (DFS) for maintenance planning."""
    start_time = time.perf_counter()
    stack = [initial_state]
    visited = {initial_state}
    parent: Dict[Tuple[bool, ...], Optional[Tuple[bool, ...]]] = {initial_state: None}
    action_taken: Dict[Tuple[bool, ...], str] = {}
    nodes_expanded = 0

    while stack:
        current = stack.pop()
        nodes_expanded += 1

        if is_goal(current, goal_state):
            states, actions = reconstruct_path(parent, action_taken, current)
            execution_time = time.perf_counter() - start_time
            return {
                "algorithm": "DFS",
                "found": True,
                "states": states,
                "actions": actions,
                "path_cost": sum(ACTIONS[a]["cost"] for a in actions),
                "nodes_expanded": nodes_expanded,
                "execution_time": execution_time
            }

        for next_state, action, cost in get_neighbors(current):
            if next_state not in visited:
                visited.add(next_state)
                parent[next_state] = current
                action_taken[next_state] = action
                stack.append(next_state)

    execution_time = time.perf_counter() - start_time
    return {
        "algorithm": "DFS",
        "found": False,
        "states": [],
        "actions": [],
        "path_cost": math.inf,
        "nodes_expanded": nodes_expanded,
        "execution_time": execution_time
    }


def heuristic(state: Tuple[bool, ...]) -> int:
    """
    Estimated remaining maintenance cost.
    Sums the exact costs of required maintenance actions that are not yet satisfied.
    Admissible and consistent since each action must be executed at least once.
    """
    cost = 0
    bearing_inspected = state[0]
    bearing_worn_confirmed = state[1]
    bearing_replaced = state[2]
    vibration_tested = state[4]
    temperature_tested = state[5]
    pump_tested = state[6]
    pump_operational = state[7]

    if not bearing_inspected:
        cost += ACTIONS["Inspect Bearing"]["cost"]
    if not bearing_worn_confirmed:
        cost += ACTIONS["Confirm Bearing Wear"]["cost"]
    if not bearing_replaced:
        cost += ACTIONS["Replace Bearing"]["cost"]
    if not vibration_tested:
        cost += ACTIONS["Run Vibration Test"]["cost"]
    if not temperature_tested:
        cost += ACTIONS["Run Temperature Test"]["cost"]
    if not pump_tested:
        cost += ACTIONS["Test Pump"]["cost"]
    if not pump_operational:
        cost += ACTIONS["Restore Pump Operation"]["cost"]

    return cost


def astar(initial_state: Tuple[bool, ...] = DEFAULT_INITIAL_STATE, goal_state: Optional[Tuple[bool, ...]] = None) -> Dict[str, Any]:
    """A* Search algorithm using priority queue and heuristic."""
    start_time = time.perf_counter()
    counter = 0
    initial_f = heuristic(initial_state)

    priority_queue: List[Tuple[int, int, Tuple[bool, ...]]] = [
        (initial_f, counter, initial_state)
    ]
    g_cost: Dict[Tuple[bool, ...], int] = {initial_state: 0}
    parent: Dict[Tuple[bool, ...], Optional[Tuple[bool, ...]]] = {initial_state: None}
    action_taken: Dict[Tuple[bool, ...], str] = {}
    visited = set()
    nodes_expanded = 0

    while priority_queue:
        f_score, _, current = heapq.heappop(priority_queue)

        if current in visited:
            continue
        visited.add(current)
        nodes_expanded += 1

        if is_goal(current, goal_state):
            states, actions = reconstruct_path(parent, action_taken, current)
            execution_time = time.perf_counter() - start_time
            return {
                "algorithm": "A*",
                "found": True,
                "states": states,
                "actions": actions,
                "path_cost": g_cost[current],
                "nodes_expanded": nodes_expanded,
                "execution_time": execution_time
            }

        for next_state, action, action_cost in get_neighbors(current):
            new_g_cost = g_cost[current] + action_cost
            if next_state not in g_cost or new_g_cost < g_cost[next_state]:
                g_cost[next_state] = new_g_cost
                parent[next_state] = current
                action_taken[next_state] = action
                counter += 1
                f_score = new_g_cost + heuristic(next_state)
                heapq.heappush(priority_queue, (f_score, counter, next_state))

    execution_time = time.perf_counter() - start_time
    return {
        "algorithm": "A*",
        "found": False,
        "states": [],
        "actions": [],
        "path_cost": math.inf,
        "nodes_expanded": nodes_expanded,
        "execution_time": execution_time
    }


def validate_path(
    initial_state: Tuple[bool, ...],
    actions: List[str],
    goal_state: Optional[Tuple[bool, ...]] = None
) -> Tuple[bool, str]:
    """
    Validates a sequence of actions by stepping through state transitions.
    Returns (is_valid, message).
    """
    state = initial_state
    for action in actions:
        if not is_valid_action(state, action):
            return False, f"Invalid action: '{action}' cannot be executed from state {state}"
        state = apply_action(state, action)

    if not is_goal(state, goal_state):
        return False, "Path does not reach goal conditions."

    return True, "Valid path reaching all goal conditions."


def run_all_algorithms(
    initial_state: Tuple[bool, ...] = DEFAULT_INITIAL_STATE,
    goal_state: Optional[Tuple[bool, ...]] = None
) -> Dict[str, Dict[str, Any]]:
    """Runs BFS, DFS, and A* on the specified initial and goal state."""
    return {
        "BFS": bfs(initial_state, goal_state),
        "DFS": dfs(initial_state, goal_state),
        "A*": astar(initial_state, goal_state)
    }


def format_state_for_display(state: Tuple[bool, ...]) -> Dict[str, bool]:
    """Converts a boolean state tuple into a human-readable dictionary."""
    return {name: state[i] for i, name in enumerate(STATE_VARIABLES)}
