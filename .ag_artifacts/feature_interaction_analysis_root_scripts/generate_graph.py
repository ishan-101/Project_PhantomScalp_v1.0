import networkx as nx
import matplotlib.pyplot as plt

def generate_graph():
    G = nx.DiGraph()

    # Modules
    modules = ["time_cycle", "microstructure", "options_features"]
    
    # Consumers and their dependencies
    # Arrow means "Depends on" or "Uses"
    dependencies = {
        "scripts/prepare_labels_dataset.py": ["time_cycle", "microstructure", "options_features"],
        "scripts/test_labels.py": ["time_cycle", "microstructure", "options_features"],
        "scripts/test_all_features.py": ["time_cycle", "microstructure", "options_features"],
        "app/orchestrator/backtest_v02.py": ["time_cycle", "microstructure", "options_features"],
        "app/ml/labels/regime.py": ["time_cycle"],
        "app/ml/labels/reversal.py": ["time_cycle", "microstructure"],
        "scripts/validate_labels_dataset.py": ["Merged Dataset"],
        "scripts/train_smoke.py": ["Merged Dataset"],
    }

    # Add nodes
    for m in modules:
        G.add_node(m, color="skyblue", node_type="module")
    
    G.add_node("Merged Dataset", color="lightgreen", node_type="artifact")
    
    # Add dependency edges (Consumer -> Producer)
    for consumer, targets in dependencies.items():
        consumer_name = consumer.split("/")[-1]
        G.add_node(consumer_name, color="lightgrey", node_type="consumer")
        for target in targets:
            G.add_edge(consumer_name, target)
            
    # Link prepare to merged dataset (Producer -> Artifact)
    G.add_edge("prepare_labels_dataset.py", "Merged Dataset")

    # Draw
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=2.0, iterations=100, seed=42)
    
    colors = [G.nodes[n].get("color", "white") for n in G.nodes]
    
    nx.draw(G, pos, 
            with_labels=True, 
            node_color=colors, 
            node_size=3000, 
            font_size=9, 
            font_weight='bold', 
            arrowsize=20, 
            edge_color='gray',
            width=1.5)
            
    plt.title("Feature Module Interaction Graph\n(Arrow = Depends On / Uses)", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("feature_module_dependency_graph.png", dpi=150)
    print("Graph generated: feature_module_dependency_graph.png")

if __name__ == "__main__":
    try:
        generate_graph()
    except ImportError as e:
        print(f"Error: {e}. Please ensure networkx and matplotlib are installed.")
    except Exception as e:
        print(f"An error occurred: {e}")
