from symtable import Class

import numpy as np
import pandas as pd

class CausalNode:
    def __init__(self,name,distribution="normal",parents=None,params=None):
        self.name = name
        self.distribution = distribution
        self.parents = parents or []
        self.params = params or {}
        self.data =None

    def sample(self, n_samples):
        if self.distribution == "categorical":
            # Probability-based selection for strings
            choices = self.params.get("choices", [])
            probs = self.params.get("probs", None)
            self.data = np.random.choice(choices, size=n_samples, p=probs)

        elif self.distribution == "normal":
            if not self.parents:
                # Root numeric node
                self.data = np.random.normal(
                    self.params.get("mean", 0),
                    self.params.get("std", 1),
                    n_samples
                )
            else:
                # Dependent numeric node with Hybrid Parent support
                intercept = self.params.get("intercept", 0)
                total_effect = intercept

                for parent in self.parents:
                    # Check if parent is numeric or categorical
                    if isinstance(parent.data[0], (int, float, np.number)):
                        slope = self.params.get(f"slope_{parent.name}", 1)
                        total_effect += (slope * parent.data)
                    else:
                        # Discrete Mapping (The 'Categorical Slope')
                        mapping = self.params.get(f"map_{parent.name}", {})
                        effect_multiplier = np.array([mapping.get(val, 1.0) for val in parent.data])
                        total_effect *= effect_multiplier

                noise_std = self.params.get("noise", 0.1)
                self.data = total_effect + np.random.normal(0, noise_std, n_samples)

        return self.data

class CausalGraph:
    def __init__(self):
        self.nodes = {}

    def add_node(self,node):
        self.nodes[node.name] = node

    def _get_execution_order(self):
        ordered_nodes = []
        visited = set()
        visiting = set()

        def visit(node):
            # 1. Eğer düğüm zaten 'visiting' içindeyse, bir döngü bulduk demektir!
            if node.name in visiting:
                raise ValueError(
                    f"Causal Cycle Detected! Node '{node.name}' depends on itself "
                    f"through its ancestry. Graphs must be Directed Acyclic (DAG)."
                )

            if node.name not in visited:
                # Düğümü işlem yoluna ekle
                visiting.add(node.name)

                for parent in node.parents:
                    visit(parent)

                # İşlem bittiğinde işlem yolundan çıkar ve 'tamamlandı'ya ekle
                visiting.remove(node.name)
                visited.add(node.name)
                ordered_nodes.append(node)

        for node in self.nodes.values():
            visit(node)

        return ordered_nodes

    def generate(self, n_samples):
        # Önce topolojik sırayı alıyoruz, eğer döngü varsa burada ValueError fırlayacak
        execution_order = self._get_execution_order()
        results = {}

        for node in execution_order:
            results[node.name] = node.sample(n_samples)

        return pd.DataFrame(results)


if __name__ == "__main__":
    factory = CausalGraph()

    # Kısır Döngü Senaryosu: A -> B -> A
    node_a = CausalNode("node_a")
    node_b = CausalNode("node_b", parents=[node_a])

    # Hata burada: A'yı B'nin ebeveyni yaptık, şimdi B'yi de A'nın ebeveyni yapıyoruz
    node_a.parents = [node_b]

    factory.add_node(node_a)
    factory.add_node(node_b)

    try:
        # Bu satır ValueError fırlatmalı!
        df = factory.generate(100)
    except ValueError as e:
        print(f"Success! Caught expected error: {e}")