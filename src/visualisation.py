import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

print("Loading data...")

with open("embeddings.json", "r") as f:
    embeddings_list = json.load(f)

embeddings_matrix = np.array(embeddings_list)
pca = PCA(n_components=2)
coords_2d = pca.fit_transform(embeddings_matrix)

id_to_party = {}
with open("data/raw/all_votes_person_id.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        person_id = int(row["person_id"])
        party = row["Fraktion/Gruppe"]
        id_to_party[person_id] = party

party_colors = {
    "SPD": "crimson",
    "CDU/CSU": "dimgray",
    "FDP": "gold",
    "AfD": "dodgerblue",
    "DIE LINKE": "hotpink",
    "DIE LINKE.": "hotpink",
    "BÜ90/GR": "forestgreen",
    "BÜNDNIS`90/DIE GRÜNEN": "forestgreen",
    "Fraktionslos": "lightgray",
    "fraktionslose": "lightgray"
}

x_coords = coords_2d[:, 0] 
y_coords = coords_2d[:, 1] 

colors_list = []
for i in range(len(embeddings_list)):
    party_name = id_to_party.get(i, "Unknown")
    colors_list.append(party_colors.get(party_name, "lightgray"))

plt.figure(figsize=(10, 8)) 
plt.scatter(x_coords, y_coords, c=colors_list, alpha=0.7, edgecolors='black', linewidth=0.5)

plt.title("Ideological Map of the Bundestag (PCA Projection)")
plt.xlabel("Axe 1 (Principal)")
plt.ylabel("Axe 2 (Secondaire)")

plt.show()