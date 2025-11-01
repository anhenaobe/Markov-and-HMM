import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from hmmlearn import hmm

# -----------------------------
# 1. Definir el modelo HMM
# -----------------------------
states = ["Positivo", "Neutro", "Negativo"]
n_states = len(states)

# Modelo HMM de ejemplo (3 estados ocultos)
model = hmm.MultinomialHMM(n_components=n_states, n_iter=100, random_state=42)

# Matriz de transición
trans_mat = np.array([
    [0.6, 0.3, 0.1],
    [0.2, 0.5, 0.3],
    [0.1, 0.2, 0.7]
])
model.transmat_ = trans_mat

# Distribución inicial
model.startprob_ = np.array([0.5, 0.3, 0.2])

# Matriz de emisión 
model.emissionprob_ = np.array([
    [0.6, 0.4],
    [0.5, 0.5],
    [0.3, 0.7]
])

# -----------------------------
# 2. Graficar el diagrama de estados
# -----------------------------
G = nx.DiGraph()
for i, from_state in enumerate(states):
    for j, to_state in enumerate(states):
        G.add_edge(from_state, to_state, weight=model.transmat_[i, j])

pos = nx.circular_layout(G)
nx.draw(G, pos, with_labels=True, node_size=2500, node_color='skyblue', font_size=10, arrowsize=20)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels={(i, j): f"{p:.2f}" for (i, j), p in labels.items()})
plt.title("Diagrama de Transiciones de Estados (HMM de Sentimientos)")

# Guardar el diagrama
plt.savefig("hmm_states.png", dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------
# 3. Graficar la matriz de transición
# -----------------------------
print("Matriz de transición:")
print(model.transmat_)

plt.figure(figsize=(6, 4))
sns.heatmap(model.transmat_, annot=True, cmap="Blues",
            xticklabels=states, yticklabels=states, fmt=".2f")
plt.title("Matriz de Transición de Estados (HMM de Sentimientos)")

# Guardar la matriz
plt.savefig("hmm_matrix.png", dpi=300, bbox_inches='tight')
plt.show()
