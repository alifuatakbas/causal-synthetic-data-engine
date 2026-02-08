import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


class CausalNode:
    def __init__(self, name, distribution="normal", parents=None, params=None, constraints=None):
        self.name = name
        self.distribution = distribution
        self.parents = parents or []
        self.params = params or {}

        self.constraints = constraints or {}
        self.data = None

    def _apply_constraints(self):
        """Üretilen veriye fiziksel sınırları uygular."""
        if self.data is None: # if x=0 --> undefined
            return

        ## Clamp min to max
        minimum = self.constraints.get("min")
        maximum = self.constraints.get("max")

        if minimum is not None or maximum is not None:
            self.data = np.clip(self.data, a_min=minimum, a_max=maximum)


        decimals = self.constraints.get("decimals")
        if decimals is not None:
            self.data = np.round(self.data, decimals=decimals)

    def sample(self, n_samples):
        if self.distribution == "categorical":
            choices = self.params.get("choices", [])
            probs = self.params.get("probs", None)
            self.data = np.random.choice(choices, size=n_samples, p=probs)

        elif self.distribution == "normal":
            if not self.parents:
                self.data = np.random.normal(self.params.get("mean", 0), self.params.get("std", 1), n_samples)
            else:
                intercept = self.params.get("intercept", 0)
                total_effect = intercept

                for parent in self.parents:
                    # Ebeveyn verisini al
                    parent_val = parent.data

                    # --- LOCAL TRANSFORM BURADA ---
                    # Her ebeveyne özel bir transform tanımlayabiliriz
                    # Örn: params={'transform_age': 'square'}
                    p_transform = self.params.get(f"transform_{parent.name}")

                    if p_transform == "square":
                        parent_val = np.power(parent_val, 2)
                    elif p_transform == "sqrt":
                        parent_val = np.sqrt(np.abs(parent_val))
                    elif p_transform == "log":
                        parent_val = np.log(np.abs(parent_val) + 1e-6)

                    # Etkiyi hesapla ve toplam etkiye ekle/çarp
                    if isinstance(parent.data[0], (int, float, np.number)):
                        slope = self.params.get(f"slope_{parent.name}", 1)
                        total_effect += (slope * parent_val)
                    else:
                        mapping = self.params.get(f"map_{parent.name}", {})
                        effect_multiplier = np.array([mapping.get(val, 1.0) for val in parent.data])
                        total_effect *= effect_multiplier

                # Gürültü ekle ve veriyi oluştur
                noise_std = self.params.get("noise", 0.1)
                self.data = total_effect + np.random.normal(0, noise_std, n_samples)

        self._apply_constraints()
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

    def export_to_csv(self,filename="casualtwin_data.csv"):

        #Check
        first_node = next(iter(self.nodes.values()))
        if first_node.data is None:
            raise ValueError("No data generated yet!")
        results = {name: node.data for name, node in self.nodes.items()}
        df = pd.DataFrame(results)
        df.to_csv(filename,index=False)
        print(f"Success: Data exported to {filename}")
        return df
    def plot_all(self):
        #Check
        df = pd.DataFrame({name: node.data for name, node in self.nodes.items() if node.data is not None})
        if df.empty:
            raise ValueError("No data generated yet!")

        fig = plt.figure(figsize=(16, 6))

        ax1 = fig.add_subplot(121)
        G = nx.DiGraph()
        for node in self.nodes.values():
            G.add_node(node.name)
            for parent in node.parents:
                G.add_edge(parent.name, node.name)
        pos = nx.spring_layout(G)
        nx.draw(G,pos,with_labels=True,node_color ='skyblue',node_size=2000,arrowsize=20,font_size=12,font_weight='bold',ax=ax1)
        ax1.set_title("Causal Relationship Graph (DAG)")

        # --- SAĞ PANEL: VERİ DAĞILIMLARI ---
        ax2 = fig.add_subplot(122)
        # Sadece sayısal sütunları görselleştirelim
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            # Verileri normalize edip yan yana histogram/kde çizelim
            for i, col in enumerate(numeric_df.columns):
                sns.kdeplot(df[col], fill=True, label=col, ax=ax2)

        ax2.set_title("Data Distributions (KDE Plots)")
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def analyze_sensitivity(self, input_node_name, target_node_name, n_samples=5000, perturbation=0.1):
        """
        Input node'u 'perturbation' kadar değiştirir ve target node üzerindeki etkiyi ölçer.
        """
        # 1. Senaryo: Referans (Base) Veri Üretimi
        df_base = self.generate(n_samples)
        base_mean_input = df_base[input_node_name].mean()
        base_mean_target = df_base[target_node_name].mean()

        # 2. Senaryo: Müdahale (Perturbation)
        # Input node'un ortalamasını %X kadar 'dürtüyoruz'
        input_node = self.nodes[input_node_name]
        original_mean = input_node.params.get("mean", 0)

        # Ortalama değeri geçici olarak değiştir
        input_node.params["mean"] = original_mean * (1 + perturbation)

        # Yeni veriyi üret
        df_perturbed = self.generate(n_samples)
        new_mean_input = df_perturbed[input_node_name].mean()
        new_mean_target = df_perturbed[target_node_name].mean()

        # Eski değeri geri yükle (Sistemi bozmamak için)
        input_node.params["mean"] = original_mean

        # 3. Hesaplama: Duyarlılık Skoru
        delta_x = (new_mean_input - base_mean_input) / (base_mean_input + 1e-6)
        delta_y = (new_mean_target - base_mean_target) / (base_mean_target + 1e-6)

        sensitivity_score = delta_y / (delta_x + 1e-6)

        return {
            "input": input_node_name,
            "target": target_node_name,
            "sensitivity_score": round(sensitivity_score, 4),
            "impact_direction": "Positive" if sensitivity_score > 0 else "Negative"
        }


if __name__ == "__main__":
    factory = CausalGraph()

    # Yaş: Ortalama 35
    age_node = CausalNode("age", params={"mean": 35, "std": 10},
                          constraints={"min": 18, "max": 65, "decimals": 0})

    # Eğitim: %40 Lise, %60 Üniversite
    edu_node = CausalNode("education", distribution="categorical",
                          params={"choices": ["High School", "University"], "probs": [0.4, 0.6]})

    # Gelir: Base 10.000 TL + (Yaş^2 * 5) + Eğitim Çarpanı
    # Artık intercept (10.000) karesi alınmadığı için güvende!
    income_node = CausalNode("income", parents=[age_node, edu_node], params={
        "intercept": 2000,
        "slope_age": 0.1,
        "transform_age": "square", # Sadece yaşın karesini al
        "map_education": {"High School": 1.0, "University": 1.3},
        "noise": 200
    }, constraints={"min": 2000, "decimals": 2})

    factory.add_node(age_node)
    factory.add_node(edu_node)
    factory.add_node(income_node)

    df = factory.generate(1000)
    print(df.head())
    print("\nOrtalama Gelir:", df['income'].mean())

    # Veriyi CSV olarak dışa aktar
    df_exported = factory.export_to_csv("my_synthetic_income_data.csv")

    # Dosyanın kaydedildiğini doğrulamak için ilk birkaç satırı oku
    print("\n--- CSV EXPORT TEST ---")
    print(f"File saved with {len(df_exported)} rows.")