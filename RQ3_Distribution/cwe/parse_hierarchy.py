"""
Deprecated script to parse the CWE hierarchy tree from the XML file.

Use the script in "./CWE-1000-circle-tree-visual" instead.
"""

import json 
import xml.etree.ElementTree as ET
from collections import deque


def build_hierarchy_from_second_layer(xml_root, second_layer_ids):
    """
    Builds the hierarchy tree starting from the specified second-layer CWE IDs.
    """
    hierarchy_tree = {"1000": {cwe_id: {} for cwe_id in second_layer_ids}}
    queue = deque(second_layer_ids)

    parent_child_map = {}

    # Extract parent-child relationships from XML
    for weakness in xml_root.find("{http://cwe.mitre.org/cwe-7}Weaknesses"):
        cwe_id = weakness.get("ID")
        related_weaknesses = weakness.find("{http://cwe.mitre.org/cwe-7}Related_Weaknesses")
        
        if related_weaknesses is not None:
            for rel_weakness in related_weaknesses:
                if rel_weakness.get("Nature") == "ChildOf":
                    parent_id = rel_weakness.get("CWE_ID")
                    if parent_id not in parent_child_map:
                        parent_child_map[parent_id] = []
                    parent_child_map[parent_id].append(cwe_id)

    # BFS to build the tree
    while queue:
        current_node = queue.popleft()
        children = parent_child_map.get(current_node, [])
        for child in children:
            # Insert child into the tree
            current_layer = hierarchy_tree["1000"]
            for key in current_node.split("-"):
                current_layer = current_layer.setdefault(key, {})
            current_layer[child.split("-")[-1]] = {}
            # Add child to the queue
            queue.append(child)

    return hierarchy_tree

# Second layer CWE IDs
second_layer_ids = ["284", "435", "664", "682", "693", "703", "707", "710"]

root = ET.parse('/home/kaixuan/patch_locating/cwe/1000.xml/1000.xml').getroot()

# Rebuild the complete hierarchy tree from the second layer
complete_hierarchy_tree_second_layer = build_hierarchy_from_second_layer(root, second_layer_ids)

# Save the complete hierarchy tree as a JSON file starting from the second layer
complete_json_file_path_second_layer = './complete_cwe_hierarchy_tree_second_layer.json'
with open(complete_json_file_path_second_layer, 'w') as json_file:
    json.dump(complete_hierarchy_tree_second_layer, json_file, indent=4)

complete_json_file_path_second_layer


xml_file_path = './1000.xml'

import networkx as nx
import matplotlib.pyplot as plt

def draw_hierarchy_tree(hierarchy_tree):
    """
    Draws the hierarchy tree using NetworkX.
    """
    G = nx.DiGraph()

    def add_nodes_edges(current_node, parent_node=None):
        for node, children in current_node.items():
            G.add_node(node)
            if parent_node:
                G.add_edge(parent_node, node)
            add_nodes_edges(children, node)

    add_nodes_edges(hierarchy_tree)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray', linewidths=1, font_size=10)
    plt.title("CWE Hierarchy Tree (Starting from Second Layer)")
    plt.show()
    plt.savefig("./CWE_Hierarchy_Tree_Second_Layer.png")

# Draw the hierarchy tree
draw_hierarchy_tree(complete_hierarchy_tree_second_layer)
