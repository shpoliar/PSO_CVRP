import xml.etree.ElementTree as ET
import numpy as np

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
        "vehicles": 7,  # з назви A-n54-k07
        "distance_matrix": distance_matrix,
        "demands": demands,
        "depot": depot,
        "coordinates": nodes  # додаємо координати!
    }

