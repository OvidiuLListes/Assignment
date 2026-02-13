import math
from typing import List, Tuple, Dict, Any


# ----------------------------
# Data structures
# ----------------------------

class Event:
    def __init__(self, x: float, y: float, capacity: float, event_type: str, name: str):
        self.x = x
        self.y = y
        self.capacity = capacity
        self.type = event_type  # "delivery", "pickup", or "depot"
        self.name = name

    def coord(self):
        return (self.x, self.y)

    def __repr__(self):
        return f"{self.name}({self.type}, cap={self.capacity}, ({self.x},{self.y}))"


DEPOT = Event(0.0, 0.0, 0.0, "depot", "Depot")


# ----------------------------
# Distance utilities
# ----------------------------

def distance(a: Event, b: Event) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def route_length(route: List[Event]) -> float:
    total = 0.0
    for i in range(len(route) - 1):
        total += distance(route[i], route[i + 1])
    return total


# ----------------------------
# Delivery selection (knapsack-style greedy)
# ----------------------------

def select_deliveries(deliveries: List[Event], capacity_limit: float) -> List[Event]:
    """
    Greedy heuristic:
    prioritize small capacity and proximity to depot.
    """
    deliveries_sorted = sorted(
        deliveries,
        key=lambda d: (d.capacity, distance(DEPOT, d))
    )

    selected = []
    total_capacity = 0.0

    for d in deliveries_sorted:
        if total_capacity + d.capacity <= capacity_limit:
            selected.append(d)
            total_capacity += d.capacity

    return selected


# ----------------------------
# Route construction
# ----------------------------

def nearest_neighbor_route(deliveries: List[Event]) -> List[Event]:
    """
    Build an initial route:
    depot -> deliveries (nearest neighbor order)
    """
    route = [DEPOT]
    current = DEPOT
    unvisited = deliveries.copy()

    while unvisited:
        next_event = min(unvisited, key=lambda d: distance(current, d))
        route.append(next_event)
        current = next_event
        unvisited.remove(next_event)

    return route


def insert_pickup_best_position(route: List[Event], pickup: Event) -> List[Event]:
    """
    Insert pickup into route position that minimizes extra distance.
    """
    best_position = None
    best_increase = float("inf")

    for i in range(1, len(route)):
        increase = (
            distance(route[i - 1], pickup)
            + distance(pickup, route[i])
            - distance(route[i - 1], route[i])
        )
        if increase < best_increase:
            best_increase = increase
            best_position = i

    route.insert(best_position, pickup)
    return route


def two_opt(route: List[Event]) -> List[Event]:
    """
    2-opt local optimization for route shortening.
    """
    improved = True
    best_route = route
    best_length = route_length(route)

    while improved:
        improved = False
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                if j - i == 1:
                    continue

                new_route = best_route[:]
                new_route[i:j] = reversed(best_route[i:j])

                new_length = route_length(new_route)
                if new_length < best_length:
                    best_route = new_route
                    best_length = new_length
                    improved = True

        route = best_route

    return best_route


def build_route(deliveries: List[Event], pickup: Event) -> List[Event]:
    """
    Full route:
    depot -> deliveries -> pickup -> depot
    optimized.
    """
    route = nearest_neighbor_route(deliveries)
    route = insert_pickup_best_position(route, pickup)
    route.append(DEPOT)
    route = two_opt(route)
    return route


# ----------------------------
# Capacity feasibility check
# ----------------------------

def is_capacity_feasible(route: List[Event], vehicle_capacity: float) -> bool:
    """
    Validate load across the route.
    Start with total delivery load.
    Deliveries decrease load; pickup increases.
    """
    total_delivery_load = sum(e.capacity for e in route if e.type == "delivery")
    current_load = total_delivery_load

    if current_load > vehicle_capacity:
        return False

    for e in route:
        if e.type == "delivery":
            current_load -= e.capacity
        elif e.type == "pickup":
            current_load += e.capacity

        if current_load > vehicle_capacity:
            return False

    return True


# ----------------------------
# Main solver
# ----------------------------

def solve(deliveries: List[Event], pickups: List[Event], vehicle_capacity: float) -> Dict[str, Any]:
    """
    Try each pickup, maximize deliveries, then minimize route length.
    """
    best_solution = None
    best_score = (-1, float("inf"))

    for pickup in pickups:

        capacity_for_deliveries = vehicle_capacity # - pickup.capacity
        if capacity_for_deliveries < 0:
            continue

        selected_deliveries = select_deliveries(deliveries, capacity_for_deliveries)
        route = build_route(selected_deliveries, pickup)

        if not is_capacity_feasible(route, vehicle_capacity):
            continue

        num_deliveries = len(selected_deliveries)
        length = route_length(route)

        score = (num_deliveries, -length)

        if score > best_score:
            best_score = score
            best_solution = {
                "route": route,
                "num_deliveries": num_deliveries,
                "route_length": length,
                "pickup_used": pickup
            }

    return best_solution


# ----------------------------
# Example usage
# ----------------------------

if __name__ == "__main__":

    # Example deliveries
    deliveries = [
        Event(2, 3, 2, "delivery", "D1"),
        Event(5, 4, 1, "delivery", "D2"),
        Event(1, 7, 3, "delivery", "D3"),
        Event(6, 1, 2, "delivery", "D4"),
        Event(8, 9, 4, "delivery", "D5"),
    ]

    # Example pickups
    pickups = [
        Event(4, 6, 4, "pickup", "P1"),
        Event(7, 2, 3, "pickup", "P2"),
        Event(3, 8, 1, "pickup", "P3"),
    ]

    vehicle_capacity = 8

    solution = solve(deliveries, pickups, vehicle_capacity)

    if solution:
        print("Best route found:\n")
        for e in solution["route"]:
            print(e)

        print("\nPickup chosen:", solution["pickup_used"].name)
        print("Deliveries served:", solution["num_deliveries"])
        print("Route length:", round(solution["route_length"], 2))
    else:
        print("No feasible route found.")
