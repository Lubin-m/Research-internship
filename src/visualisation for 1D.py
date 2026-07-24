import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from preprocessing import load_and_prepare_graph

print("Loading data...")

# Load Embeddings
json_path = "/content/Research-internship/embeddings.json"
with open(json_path, "r") as f:
    embeddings_list = json.load(f)

embeddings_matrix = np.array(embeddings_list)

# Map person_id to their political party
id_to_party = {}
with open("data/raw/all_votes_person_id.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        person_id = int(row["person_id"])
        party = row["Fraktion/Gruppe"]
        if "LINKE" in party: party = "DIE LINKE"
        elif "GRÜN" in party or "BÜ90" in party: party = "GRÜNE"
        elif "CDU" in party or "CSU" in party: party = "CDU/CSU"
        elif "fraktionslos" in party.lower(): party = "Fraktionslos"
        id_to_party[person_id] = party

print("Extraction directe de l'espace idéologique 1D du GNN...")

# 1. On récupère directement l'unique dimension générée par le modèle
# emb[0] car chaque embedding est maintenant une liste d'un seul élément [x]
x_coords = np.array([emb[0] for emb in embeddings_matrix])

# 2. Vérification de l'orientation (on veut la Gauche à gauche et la Droite à droite)
afd_mean_x = np.mean([x_coords[i] for i, emb in enumerate(embeddings_matrix) if id_to_party.get(i) == "AfD"])
linke_mean_x = np.mean([x_coords[i] for i, emb in enumerate(embeddings_matrix) if id_to_party.get(i) == "DIE LINKE"])

# Si l'AfD s'est retrouvée mathématiquement avec un score plus petit que Die Linke, 
# on multiplie tout par -1 pour inverser l'axe comme un miroir.
if afd_mean_x < linke_mean_x:
    x_coords = -x_coords

# =====================================================================
# --- NOUVEAU : PARTIE BENCHMARKING (ÉVALUATION) ---
# =====================================================================
print("Chargement du graphe pour évaluation...")
graph = load_and_prepare_graph("data/processed/bundestag_signed.json")
pos_edges_np = graph.pos_edge_index.cpu().numpy()
neg_edges_np = graph.neg_edge_index.cpu().numpy()

# 1. Discrétisation en 8 intervalles
num_bins = 8
bins = np.linspace(min(x_coords), max(x_coords), num_bins + 1)
node_assignments = np.digitize(x_coords, bins) - 1
node_assignments[node_assignments == num_bins] = num_bins - 1  # Sécurité pour la valeur max

# 2. Fonction de calcul du désaccord (Disagreement)
def calculate_disagreement(pos_sources, pos_targets, neg_sources, neg_targets, assignments):
    errors = 0
    
    # Règle 1 : Liens Positifs (+) -> Doivent se chevaucher (distance <= 1)
    for u, v in zip(pos_sources, pos_targets):
        if abs(assignments[u] - assignments[v]) > 1:
            errors += 1
            
    # Règle 2 : Liens Négatifs (-) -> Doivent être disjoints (distance > 1)
    for u, v in zip(neg_sources, neg_targets):
        if abs(assignments[u] - assignments[v]) <= 1:
            errors += 1
            
    total_edges = len(pos_sources) + len(neg_sources)
    error_rate = (errors / total_edges) * 100 if total_edges > 0 else 0
    
    return errors, total_edges, error_rate

# 3. Calcul et affichage
erreurs, total_aretes, score_final = calculate_disagreement(
    pos_edges_np[0], pos_edges_np[1], 
    neg_edges_np[0], neg_edges_np[1], 
    node_assignments
)

print("\n=== RÉSULTATS DU BENCHMARK ===")
print(f"Total des arêtes évaluées : {total_aretes}")
print(f"Nombre d'arêtes violées (Disagreement) : {erreurs}")
print(f"Score d'erreur de ton GNN : {score_final:.2f}%\n")
# =====================================================================


# Define the specific layout
party_order = ["DIE LINKE", "GRÜNE", "SPD", "FDP", "CDU/CSU", "AfD"]
party_colors = {
    "DIE LINKE": "hotpink",
    "GRÜNE": "forestgreen",
    "SPD": "crimson",
    "FDP": "gold",
    "CDU/CSU": "dimgray",
    "AfD": "dodgerblue"
}

party_data = {party: [] for party in party_order}
for i, x in enumerate(x_coords):
    party = id_to_party.get(i, "Unknown")
    if party in party_data:
        party_data[party].append(x)

# Create the plot
fig, ax = plt.subplots(figsize=(14, 6))

x_min, x_max = min(x_coords) - 0.1, max(x_coords) + 0.1
bands = np.linspace(x_min, x_max, 9) 
for i in range(len(bands) - 1):
    if i % 2 == 0:
        ax.axvspan(bands[i], bands[i+1], facecolor='lightgray', alpha=0.3, zorder=0)

for y_index, party in enumerate(party_order, start=1):
    xs = np.array(party_data[party])
    if len(xs) == 0:
        continue
    
    color = party_colors[party]
    ax.hlines(y=y_index, xmin=min(xs), xmax=max(xs), color=color, linewidth=2, zorder=1)
    
    jitter = np.random.normal(0, 0.15, size=len(xs))
    ax.scatter(xs, y_index + jitter, c=color, edgecolors='black', linewidth=0.5, alpha=0.7, s=30, zorder=2)

# Styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.set_yticks([]) 
ax.set_xlabel("Political Spectrum (Anchored Axis Projection)", fontsize=12, labelpad=10)

legend_patches = [mpatches.Patch(color=party_colors[p], label=p) for p in party_order]
ax.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
          ncol=3, frameon=False, fontsize=11)

plt.tight_layout()
plt.show()