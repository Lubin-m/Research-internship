import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

print("Loading data...")

# 1. Load Embeddings
with open("embeddings.json", "r") as f:
    embeddings_list = json.load(f)

embeddings_matrix = np.array(embeddings_list)

# 2. Map person_id to their political party
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

print("Calcul de l'axe polarisé Gauche-Droite...")

# --- LA SOLUTION : PROJECTION VECTORIELLE ---
# On isole les positions brutes des deux extrêmes
afd_embeds = [embeddings_matrix[i] for i in range(len(embeddings_list)) if id_to_party.get(i) == "AfD"]
linke_embeds = [embeddings_matrix[i] for i in range(len(embeddings_list)) if id_to_party.get(i) == "DIE LINKE"]

# On calcule leur centre de gravité respectif
afd_center = np.mean(afd_embeds, axis=0)
linke_center = np.mean(linke_embeds, axis=0)

# On crée le vecteur "Axe Idéologique" qui va de Die Linke vers l'AfD
lr_axis = afd_center - linke_center
lr_axis_normalized = lr_axis / np.linalg.norm(lr_axis)

# On projette chaque député sur cet axe spécifique (Produit scalaire)
x_coords = []
for emb in embeddings_matrix:
    # Projection = (Position_député - Pôle_Gauche) * Vecteur_Normalisé
    proj = np.dot(emb - linke_center, lr_axis_normalized)
    x_coords.append(proj)

x_coords = np.array(x_coords)
# ---------------------------------------------

# 3. Define the specific layout
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

# 4. Create the plot
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

# 5. Styling
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