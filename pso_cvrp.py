import numpy as np
import random
import datetime
import xml.etree.ElementTree as ET
from sklearn.cluster import KMeans


def time_of_function(function):
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        result = function(*args, **kwargs)
        used_time = datetime.datetime.now() - start_time
        return result, used_time.total_seconds()
    return wrapper


def parse_xml_cvrp(path):
    tree = ET.parse(path)
    root = tree.getroot()

    nodes = []
    for node in root.findall(".//node"):
        x = float(node.find("cx").text)
        y = float(node.find("cy").text)
        nodes.append((x, y))

    dimension = len(nodes)

    distance_matrix = np.zeros((dimension, dimension))
    for i in range(dimension):
        for j in range(dimension):
            dx = nodes[i][0] - nodes[j][0]
            dy = nodes[i][1] - nodes[j][1]
            distance_matrix[i][j] = round(np.sqrt(dx**2 + dy**2))

    demands = [0] * dimension
    for req in root.findall(".//request"):
        node_id = int(req.get("node")) - 1
        quantity = float(req.find("quantity").text)
        demands[node_id] = quantity

    capacity = int(float(root.find(".//vehicle_profile/capacity").text))
    depot = int(root.find(".//vehicle_profile/departure_node").text) - 1

    return {
        "dimension": dimension,
        "capacity": capacity,
        "vehicles": 7,
        "distance_matrix": distance_matrix,
        "demands": demands,
        "depot": depot,
        "coordinates": nodes
    }


def generate_clustered_initial_solution(demands, depot, coordinates, num_vehicles):
    num_customers = len(demands) - 1
    customer_indices = [i for i in range(len(demands)) if i != depot]
    customer_coords = [coordinates[i] for i in customer_indices]

    kmeans = KMeans(n_clusters=num_vehicles, random_state=0)
    labels = kmeans.fit_predict(customer_coords)

    clustered_customers = [[] for _ in range(num_vehicles)]
    for idx, label in zip(customer_indices, labels):
        clustered_customers[label].append(idx)

    for cluster in clustered_customers:
        random.shuffle(cluster)

    clustered_permutation = [client for cluster in clustered_customers for client in cluster]
    return clustered_permutation


class ParticleCVRP:
    def __init__(self, dimension, initial_position=None):
        if initial_position is not None:
            self.position = list(initial_position)
        else:
            self.position = list(np.random.permutation(range(1, dimension)))
        self.velocity = []
        self.best_position = list(self.position)
        self.best_fitness = float('inf')


def improved_split_routes(solution, demands, capacity):
    routes = []
    current_route = []
    current_load = 0
    for customer in solution:
        demand = demands[customer]
        if current_load + demand <= capacity:
            current_route.append(customer)
            current_load += demand
        else:
            routes.append(current_route)
            current_route = [customer]
            current_load = demand
    if current_route:
        routes.append(current_route)
    return routes


def mutate_insert_customer(routes):
    routes = [r[:] for r in routes if r]
    if len(routes) < 2:
        return routes

    from_route_idx, to_route_idx = random.sample(range(len(routes)), 2)
    from_route = routes[from_route_idx]
    to_route = routes[to_route_idx]

    if not from_route:
        return routes

    customer_idx = random.randint(0, len(from_route) - 1)
    customer = from_route.pop(customer_idx)

    insert_pos = random.randint(0, len(to_route))
    to_route.insert(insert_pos, customer)

    routes[from_route_idx] = from_route
    routes[to_route_idx] = to_route
    return routes


def apply_2opt(route, distance_matrix):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 1):
            for j in range(i + 1, len(best)):
                if j - i == 1:
                    continue
                new_route = best[:i] + best[i:j][::-1] + best[j:]
                if calculate_route_cost(new_route, distance_matrix) < calculate_route_cost(best, distance_matrix):
                    best = new_route
                    improved = True
    return best


def calculate_route_cost(route, distance_matrix, depot=0):
    if not route:
        return 0
    total = distance_matrix[depot][route[0]]
    for i in range(len(route) - 1):
        total += distance_matrix[route[i]][route[i + 1]]
    total += distance_matrix[route[-1]][depot]
    return total


def apply_2opt_to_all(routes, distance_matrix, depot=0):
    return [apply_2opt(route, distance_matrix) for route in routes]


def evaluate_solution(solution, distance_matrix, demands, capacity, depot, max_routes=None):
    total_distance = 0
    routes = improved_split_routes(solution, demands, capacity)

    if max_routes and len(routes) > max_routes:
        penalty = (len(routes) - max_routes) * 1000
        total_distance += penalty

    for route in routes:
        if not route:
            continue
        total_distance += distance_matrix[depot][route[0]]
        for i in range(len(route) - 1):
            total_distance += distance_matrix[route[i]][route[i+1]]
        total_distance += distance_matrix[route[-1]][depot]
    return total_distance


def apply_velocity(position, velocity):
    position = list(position)
    for i, j in velocity:
        position[i], position[j] = position[j], position[i]
    return position


def subtract_positions(pos1, pos2):
    swaps = []
    p1 = list(pos1)
    p2 = list(pos2)
    for i in range(len(p1)):
        if p1[i] != p2[i]:
            swap_idx = p2.index(p1[i])
            swaps.append((i, swap_idx))
            p2[i], p2[swap_idx] = p2[swap_idx], p2[i]
    return swaps


@time_of_function
def pso_cvrp(distance_matrix, demands, capacity, depot, coordinates, num_particles=40, max_iter=300, num_vehicles=7):
    num_customers = len(demands) - 1

    initial_perm = generate_clustered_initial_solution(demands, depot, coordinates, num_vehicles)
    swarm = [ParticleCVRP(num_customers, initial_perm if i == 0 else None) for i in range(num_particles)]

    global_best_position = None
    global_best_fitness = float('inf')

    for t in range(max_iter):
        for particle in swarm:
            fitness = evaluate_solution(particle.position, distance_matrix, demands, capacity, depot, max_routes=num_vehicles)
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = list(particle.position)
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = list(particle.position)

        for particle in swarm:
            w = 0.9 - (0.5 * t / max_iter)
            c1 = 1.5
            c2 = 1.8
            r1 = random.random()
            r2 = random.random()

            cognitive = subtract_positions(particle.best_position, particle.position)
            social = subtract_positions(global_best_position, particle.position)

            vel = particle.velocity[:int(w * len(particle.velocity))] + cognitive[:int(c1 * r1 * len(cognitive))]+ social[:int(c2 * r2 * len(social))]

            particle.velocity = vel
            particle.position = apply_velocity(particle.position, particle.velocity)

            mutated_routes = improved_split_routes(particle.position, demands, capacity)
            mutated_routes = mutate_insert_customer(mutated_routes)
            mutated_routes = apply_2opt_to_all(mutated_routes, distance_matrix, depot)

            particle.position = [client for route in mutated_routes for client in route]

    return {
        "best_position": global_best_position,
        "best_cost": global_best_fitness
    }


parsed = parse_xml_cvrp("test_data/A-n54-k07.xml")

result, runtime = pso_cvrp(
    distance_matrix=parsed["distance_matrix"],
    demands=parsed["demands"],
    capacity=parsed["capacity"],
    depot=parsed["depot"],
    coordinates=parsed["coordinates"],
    num_particles=50,
    max_iter=700,
    num_vehicles=parsed["vehicles"]
)
print("Найкраща вартість:", result["best_cost"])
print("Найкраща перестановка:", result["best_position"])
print("Час виконання:", runtime, "секунд")

