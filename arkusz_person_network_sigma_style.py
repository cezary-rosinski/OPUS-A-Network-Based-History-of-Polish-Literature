#%% Importy

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path

try:
    from ipysigma import Sigma
except Exception:
    Sigma = None


#%% Ustawienia

DATA = Path("data")
OUT = DATA / "outputs_sigma_style"
OUT.mkdir(parents=True, exist_ok=True)

NODES_PATH = DATA / "person_nodes.csv"
COOC_EDGES_PATH = DATA / "person_edges_cooccurrence.csv"
DIRECTED_EDGES_PATH = DATA / "person_edges_interpretive_directed.csv"
INCIDENCE_PATH = DATA / "record_person_incidence.csv"

MIN_EDGE_WEIGHT = 1
TOP_N_FOR_PNG = 120

BETWEENNESS_SAMPLE_SIZE = 500
RANDOM_SEED = 42


#%% Funkcje pomocnicze

def check_file_exists(path: Path) -> None:
    """Sprawdza, czy plik istnieje."""
    if not path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku: {path.resolve()}\n"
            f"Sprawdź, czy plik znajduje się w katalogu: {DATA.resolve()}"
        )


def ensure_column(df: pd.DataFrame, column: str, default_value) -> pd.DataFrame:
    """
    Dodaje kolumnę, jeśli jej nie ma.

    default_value może być:
    - stałą, np. 1 albo ""
    - nazwą istniejącej kolumny, np. "weight"
    - obiektem Series
    """
    if column not in df.columns:
        if isinstance(default_value, str) and default_value in df.columns:
            df[column] = df[default_value]
        else:
            df[column] = default_value
    return df


def clean_for_graphml(value):
    """
    NetworkX GraphML nie lubi pd.NA, NaN, list i złożonych obiektów.
    Ta funkcja zamienia problematyczne wartości na typy bezpieczne.
    """
    if pd.isna(value):
        return ""
    if isinstance(value, (list, tuple, set)):
        return "; ".join(map(str, value))
    return value


def safe_write_graphml(graph: nx.Graph, path: Path) -> None:
    """
    Bezpieczny eksport GraphML: czyści atrybuty węzłów i krawędzi.
    """
    graph_to_save = graph.copy()

    for _, attrs in graph_to_save.nodes(data=True):
        for key, value in list(attrs.items()):
            attrs[key] = clean_for_graphml(value)

    for _, _, attrs in graph_to_save.edges(data=True):
        for key, value in list(attrs.items()):
            attrs[key] = clean_for_graphml(value)

    nx.write_graphml(graph_to_save, path)


def classify_person_node(labels: str) -> tuple[str, str]:
    """
    Klasyfikacja osoby według ról źródłowych.
    Zwraca: node_type, color.
    """
    labels = str(labels)

    is_author = "Author" in labels
    is_creator = "Creator" in labels
    is_index = "IndexPerson" in labels
    is_person = "Person" in labels

    if is_author and (is_creator or is_index or is_person):
        return "author_and_object", "#984ea3"
    elif is_author:
        return "author_only", "#377eb8"
    elif is_creator:
        return "creator_only", "#e41a1c"
    elif is_index:
        return "indexed_person_only", "#4daf4a"
    elif is_person:
        return "person_role_only", "#ff7f00"
    else:
        return "other", "#999999"


def aggregate_edges(
    df: pd.DataFrame,
    group_cols: list[str],
    has_role_pairs: bool = False
) -> pd.DataFrame:
    """
    Agreguje potencjalnie zdublowane krawędzie.
    """
    agg_dict = {
        "weight": "sum",
        "record_count": "sum",
        "first_year": "min",
        "last_year": "max",
    }

    if has_role_pairs and "role_pairs" in df.columns:
        agg_dict["role_pairs"] = lambda x: "; ".join(
            sorted(set(str(v) for v in x.dropna() if str(v) != ""))
        )

    if "relation_type" in df.columns and "relation_type" not in group_cols:
        agg_dict["relation_type"] = lambda x: "; ".join(
            sorted(set(str(v) for v in x.dropna() if str(v) != ""))
        )

    available_agg = {k: v for k, v in agg_dict.items() if k in df.columns}

    return (
        df.groupby(group_cols, as_index=False, dropna=False)
          .agg(available_agg)
    )


#%% 1. Wczytanie danych

check_file_exists(NODES_PATH)
check_file_exists(COOC_EDGES_PATH)
check_file_exists(DIRECTED_EDGES_PATH)

nodes_df = pd.read_csv(NODES_PATH)
cooc_df = pd.read_csv(COOC_EDGES_PATH)
directed_df = pd.read_csv(DIRECTED_EDGES_PATH)

print("nodes_df columns:", list(nodes_df.columns))
print("cooc_df columns:", list(cooc_df.columns))
print("directed_df columns:", list(directed_df.columns))


#%% 1a. Porządki: węzły

if "person_id" not in nodes_df.columns:
    raise ValueError("Brakuje wymaganej kolumny w person_nodes.csv: person_id")

nodes_df = ensure_column(nodes_df, "label", nodes_df["person_id"])
nodes_df = ensure_column(nodes_df, "name", nodes_df["person_id"])
nodes_df = ensure_column(nodes_df, "source_labels", "")
nodes_df = ensure_column(nodes_df, "record_count", 0)
nodes_df = ensure_column(nodes_df, "first_year", "")
nodes_df = ensure_column(nodes_df, "last_year", "")

nodes_df["record_count"] = pd.to_numeric(
    nodes_df["record_count"],
    errors="coerce"
).fillna(0)

node_label = dict(zip(nodes_df["person_id"], nodes_df["label"].fillna(nodes_df["person_id"])))
node_name = dict(zip(nodes_df["person_id"], nodes_df["name"].fillna(nodes_df["person_id"])))
source_labels = dict(zip(nodes_df["person_id"], nodes_df["source_labels"].fillna("")))
record_count = dict(zip(nodes_df["person_id"], nodes_df["record_count"]))
first_year = dict(zip(nodes_df["person_id"], nodes_df["first_year"].fillna("")))
last_year = dict(zip(nodes_df["person_id"], nodes_df["last_year"].fillna("")))


#%% 1b. Porządki: krawędzie współwystąpień

required_cooc = ["source", "target"]
missing_cooc = [c for c in required_cooc if c not in cooc_df.columns]

if missing_cooc:
    raise ValueError(f"Brakuje wymaganych kolumn w person_edges_cooccurrence.csv: {missing_cooc}")

cooc_df = ensure_column(cooc_df, "weight", 1)
cooc_df = ensure_column(cooc_df, "record_count", "weight")
cooc_df = ensure_column(cooc_df, "first_year", "")
cooc_df = ensure_column(cooc_df, "last_year", "")
cooc_df = ensure_column(cooc_df, "role_pairs", "")

cooc_df["weight"] = pd.to_numeric(cooc_df["weight"], errors="coerce").fillna(1)
cooc_df["record_count"] = pd.to_numeric(cooc_df["record_count"], errors="coerce").fillna(cooc_df["weight"])

cooc_df = cooc_df[
    ["source", "target", "weight", "record_count", "first_year", "last_year", "role_pairs"]
].dropna(subset=["source", "target"])

cooc_df = cooc_df[cooc_df["source"] != cooc_df["target"]]
cooc_df = cooc_df[cooc_df["weight"] >= MIN_EDGE_WEIGHT]

cooc_df = aggregate_edges(
    cooc_df,
    group_cols=["source", "target"],
    has_role_pairs=True
)

print(f"Liczba krawędzi współwystąpień po czyszczeniu: {len(cooc_df)}")


#%% 1c. Porządki: krawędzie skierowane autor → osoba

required_directed = ["source", "target"]
missing_directed = [c for c in required_directed if c not in directed_df.columns]

if missing_directed:
    raise ValueError(
        f"Brakuje wymaganych kolumn w person_edges_interpretive_directed.csv: {missing_directed}"
    )

directed_df = ensure_column(directed_df, "weight", 1)
directed_df = ensure_column(directed_df, "record_count", "weight")
directed_df = ensure_column(directed_df, "first_year", "")
directed_df = ensure_column(directed_df, "last_year", "")
directed_df = ensure_column(directed_df, "relation_type", "author_to_person")

directed_df["weight"] = pd.to_numeric(directed_df["weight"], errors="coerce").fillna(1)
directed_df["record_count"] = pd.to_numeric(directed_df["record_count"], errors="coerce").fillna(directed_df["weight"])

directed_df = directed_df[
    ["source", "target", "weight", "record_count", "first_year", "last_year", "relation_type"]
].dropna(subset=["source", "target"])

directed_df = directed_df[directed_df["source"] != directed_df["target"]]

directed_df = aggregate_edges(
    directed_df,
    group_cols=["source", "target", "relation_type"],
    has_role_pairs=False
)

print(f"Liczba krawędzi skierowanych po czyszczeniu: {len(directed_df)}")


#%% 2. Budowa grafu nieskierowanego: współwystąpienia osób

G = nx.Graph()

for _, row in tqdm(cooc_df.iterrows(), total=len(cooc_df), desc="Budowa grafu współwystąpień"):
    source = row["source"]
    target = row["target"]

    if not G.has_node(source):
        G.add_node(source)

    if not G.has_node(target):
        G.add_node(target)

    G.add_edge(
        source,
        target,
        weight=float(row["weight"]),
        record_count=float(row.get("record_count", row["weight"])),
        first_year=clean_for_graphml(row.get("first_year", "")),
        last_year=clean_for_graphml(row.get("last_year", "")),
        role_pairs=clean_for_graphml(row.get("role_pairs", ""))
    )

for node in G.nodes():
    labels = source_labels.get(node, "")

    node_type, color = classify_person_node(labels)

    G.nodes[node]["label"] = str(node_label.get(node, node)).replace("PersonName:", "")
    G.nodes[node]["name"] = str(node_name.get(node, node)).replace("PersonName:", "")
    G.nodes[node]["source_labels"] = labels
    G.nodes[node]["record_count"] = float(record_count.get(node, 0))
    G.nodes[node]["first_year"] = clean_for_graphml(first_year.get(node, ""))
    G.nodes[node]["last_year"] = clean_for_graphml(last_year.get(node, ""))
    G.nodes[node]["node_type"] = node_type
    G.nodes[node]["color"] = color

print(f"Graf współwystąpień: {G.number_of_nodes()} węzłów, {G.number_of_edges()} krawędzi")


#%% 3. Metryki sieciowe dla grafu współwystąpień

degree_dict = dict(G.degree())
weighted_degree_dict = dict(G.degree(weight="weight"))

if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
    pagerank_dict = nx.pagerank(G, alpha=0.85, weight="weight")
else:
    pagerank_dict = {n: 0 for n in G.nodes()}

if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
    if G.number_of_nodes() > BETWEENNESS_SAMPLE_SIZE:
        betweenness_dict = nx.betweenness_centrality(
            G,
            k=BETWEENNESS_SAMPLE_SIZE,
            seed=RANDOM_SEED,
            weight="weight"
        )
    else:
        betweenness_dict = nx.betweenness_centrality(G, weight="weight")
else:
    betweenness_dict = {n: 0 for n in G.nodes()}

if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
    clustering_dict = nx.clustering(G, weight="weight")
else:
    clustering_dict = {n: 0 for n in G.nodes()}

community_dict = {n: -1 for n in G.nodes()}

if G.number_of_edges() > 0:
    largest_cc_nodes = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(largest_cc_nodes).copy()

    communities = nx.algorithms.community.greedy_modularity_communities(
        G_lcc,
        weight="weight"
    )

    for i, community in enumerate(communities):
        for node in community:
            community_dict[node] = i
else:
    G_lcc = G.copy()

nx.set_node_attributes(G, degree_dict, name="degree")
nx.set_node_attributes(G, weighted_degree_dict, name="weighted_degree")
nx.set_node_attributes(G, pagerank_dict, name="pagerank")
nx.set_node_attributes(G, betweenness_dict, name="betweenness")
nx.set_node_attributes(G, clustering_dict, name="clustering")
nx.set_node_attributes(G, community_dict, name="community")


#%% 4. Eksport grafu współwystąpień i metryk

safe_write_graphml(G, OUT / "arkusz_person_cooccurrence_sigma_style.graphml")
nx.write_gexf(G, OUT / "arkusz_person_cooccurrence_sigma_style.gexf")

metrics_df = pd.DataFrame({
    "person_id": list(G.nodes()),
    "label": [G.nodes[n].get("label", "") for n in G.nodes()],
    "name": [G.nodes[n].get("name", "") for n in G.nodes()],
    "node_type": [G.nodes[n].get("node_type", "") for n in G.nodes()],
    "degree": [G.nodes[n].get("degree", 0) for n in G.nodes()],
    "weighted_degree": [G.nodes[n].get("weighted_degree", 0) for n in G.nodes()],
    "pagerank": [G.nodes[n].get("pagerank", 0) for n in G.nodes()],
    "betweenness": [G.nodes[n].get("betweenness", 0) for n in G.nodes()],
    "clustering": [G.nodes[n].get("clustering", 0) for n in G.nodes()],
    "community": [G.nodes[n].get("community", -1) for n in G.nodes()],
    "record_count": [G.nodes[n].get("record_count", 0) for n in G.nodes()],
    "first_year": [G.nodes[n].get("first_year", "") for n in G.nodes()],
    "last_year": [G.nodes[n].get("last_year", "") for n in G.nodes()],
}).sort_values(
    ["weighted_degree", "degree", "pagerank"],
    ascending=False
)

metrics_df.to_csv(
    OUT / "person_network_metrics_sigma_style.csv",
    index=False,
    encoding="utf-8-sig"
)

cooc_df.to_csv(
    OUT / "person_edges_cooccurrence_cleaned.csv",
    index=False,
    encoding="utf-8-sig"
)


#%% 5. Wizualizacja statyczna: TOP N według weighted degree

if G.number_of_nodes() > 0:
    top_nodes = metrics_df.head(TOP_N_FOR_PNG)["person_id"].tolist()
    G_top = G.subgraph(top_nodes).copy()

    plt.figure(figsize=(18, 14))

    pos = nx.spring_layout(
        G_top,
        seed=RANDOM_SEED,
        k=0.35,
        weight="weight"
    )

    sizes = [
        max(40, G_top.nodes[n].get("pagerank", 0) * 120000)
        for n in G_top.nodes()
    ]

    colors = [
        G_top.nodes[n].get("color", "#999999")
        for n in G_top.nodes()
    ]

    widths = [
        max(0.2, min(4, G_top[u][v].get("weight", 1) / 3))
        for u, v in G_top.edges()
    ]

    nx.draw_networkx_edges(
        G_top,
        pos,
        alpha=0.25,
        width=widths
    )

    nx.draw_networkx_nodes(
        G_top,
        pos,
        node_size=sizes,
        node_color=colors,
        alpha=0.9
    )

    nx.draw_networkx_labels(
        G_top,
        pos,
        labels={n: G_top.nodes[n].get("label", n) for n in G_top.nodes()},
        font_size=8
    )

    plt.title("Arkusz / PBL: sieć osób — TOP według weighted degree")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        OUT / "arkusz_person_network_top_sigma_style.png",
        dpi=300
    )
    plt.close()


#%% 6. Eksport HTML przez ipysigma: graf współwystąpień

if Sigma is not None:
    try:
        Sigma.write_html(
            G,
            str(OUT / "arkusz_person_cooccurrence_network.html"),
            fullscreen=True,
            node_color="node_type",
            node_size="pagerank",
            node_label="label",
            node_label_size="weighted_degree",
            edge_size="weight"
        )
        print("Zapisano HTML:", OUT / "arkusz_person_cooccurrence_network.html")
    except Exception as e:
        print("Nie udało się zapisać HTML dla grafu współwystąpień przez ipysigma.")
        print("Błąd:", e)
else:
    print("ipysigma nie jest zainstalowana. Pomijam eksport HTML Sigma dla grafu współwystąpień.")


#%% 7. Budowa grafu skierowanego: autor → osoba powiązana

DG = nx.DiGraph()

for _, row in tqdm(directed_df.iterrows(), total=len(directed_df), desc="Budowa grafu skierowanego"):
    source = row["source"]
    target = row["target"]

    if not DG.has_node(source):
        DG.add_node(source)

    if not DG.has_node(target):
        DG.add_node(target)

    if DG.has_edge(source, target):
        DG[source][target]["weight"] += float(row.get("weight", 1))
        DG[source][target]["record_count"] += float(row.get("record_count", row.get("weight", 1)))
    else:
        DG.add_edge(
            source,
            target,
            weight=float(row.get("weight", 1)),
            record_count=float(row.get("record_count", row.get("weight", 1))),
            first_year=clean_for_graphml(row.get("first_year", "")),
            last_year=clean_for_graphml(row.get("last_year", "")),
            relation_type=clean_for_graphml(row.get("relation_type", "author_to_person"))
        )

for node in DG.nodes():
    labels = source_labels.get(node, "")

    node_type, color = classify_person_node(labels)

    DG.nodes[node]["label"] = str(node_label.get(node, node)).replace("PersonName:", "")
    DG.nodes[node]["name"] = str(node_name.get(node, node)).replace("PersonName:", "")
    DG.nodes[node]["source_labels"] = labels
    DG.nodes[node]["node_type"] = node_type
    DG.nodes[node]["color"] = color

print(f"Graf skierowany: {DG.number_of_nodes()} węzłów, {DG.number_of_edges()} krawędzi")


#%% 8. Metryki sieci skierowanej

in_degree_dict = dict(DG.in_degree())
out_degree_dict = dict(DG.out_degree())
weighted_in_degree_dict = dict(DG.in_degree(weight="weight"))
weighted_out_degree_dict = dict(DG.out_degree(weight="weight"))
degree_dict = dict(DG.degree())

if DG.number_of_nodes() > 0 and DG.number_of_edges() > 0:
    pagerank_dict = nx.pagerank(DG, alpha=0.85, weight="weight")
else:
    pagerank_dict = {n: 0 for n in DG.nodes()}

try:
    if DG.number_of_nodes() > 0 and DG.number_of_edges() > 0:
        hubs_dict, authorities_dict = nx.hits(
            DG,
            max_iter=1000,
            normalized=True
        )
    else:
        hubs_dict = {n: 0 for n in DG.nodes()}
        authorities_dict = {n: 0 for n in DG.nodes()}
except Exception:
    hubs_dict = {n: 0 for n in DG.nodes()}
    authorities_dict = {n: 0 for n in DG.nodes()}

nx.set_node_attributes(DG, in_degree_dict, name="in_degree")
nx.set_node_attributes(DG, out_degree_dict, name="out_degree")
nx.set_node_attributes(DG, weighted_in_degree_dict, name="weighted_in_degree")
nx.set_node_attributes(DG, weighted_out_degree_dict, name="weighted_out_degree")
nx.set_node_attributes(DG, degree_dict, name="degree")
nx.set_node_attributes(DG, pagerank_dict, name="pagerank")
nx.set_node_attributes(DG, authorities_dict, name="authority")
nx.set_node_attributes(DG, hubs_dict, name="hub")


#%% 9. Eksport grafu skierowanego i metryk

safe_write_graphml(DG, OUT / "arkusz_person_interpretive_directed_sigma_style.graphml")
nx.write_gexf(DG, OUT / "arkusz_person_interpretive_directed_sigma_style.gexf")

directed_metrics_df = pd.DataFrame({
    "person_id": list(DG.nodes()),
    "label": [DG.nodes[n].get("label", "") for n in DG.nodes()],
    "name": [DG.nodes[n].get("name", "") for n in DG.nodes()],
    "node_type": [DG.nodes[n].get("node_type", "") for n in DG.nodes()],
    "in_degree": [DG.nodes[n].get("in_degree", 0) for n in DG.nodes()],
    "out_degree": [DG.nodes[n].get("out_degree", 0) for n in DG.nodes()],
    "weighted_in_degree": [DG.nodes[n].get("weighted_in_degree", 0) for n in DG.nodes()],
    "weighted_out_degree": [DG.nodes[n].get("weighted_out_degree", 0) for n in DG.nodes()],
    "pagerank": [DG.nodes[n].get("pagerank", 0) for n in DG.nodes()],
    "authority": [DG.nodes[n].get("authority", 0) for n in DG.nodes()],
    "hub": [DG.nodes[n].get("hub", 0) for n in DG.nodes()],
}).sort_values(
    ["weighted_in_degree", "pagerank", "authority"],
    ascending=False
)

directed_metrics_df.to_csv(
    OUT / "person_directed_network_metrics_sigma_style.csv",
    index=False,
    encoding="utf-8-sig"
)

directed_df.to_csv(
    OUT / "person_edges_directed_cleaned.csv",
    index=False,
    encoding="utf-8-sig"
)


#%% 10. Eksport HTML przez ipysigma: graf skierowany

if Sigma is not None:
    try:
        Sigma.write_html(
            DG,
            str(OUT / "arkusz_person_interpretive_directed_network.html"),
            fullscreen=True,
            node_color="node_type",
            node_size="pagerank",
            node_label="label",
            node_label_size="weighted_in_degree",
            edge_size="weight"
        )
        print("Zapisano HTML:", OUT / "arkusz_person_interpretive_directed_network.html")
    except Exception as e:
        print("Nie udało się zapisać HTML dla grafu skierowanego przez ipysigma.")
        print("Błąd:", e)
else:
    print("ipysigma nie jest zainstalowana. Pomijam eksport HTML Sigma dla grafu skierowanego.")


#%% 11. Krótkie rankingi do kontroli

print("\nTOP 20 — graf współwystąpień według weighted_degree")
print(
    metrics_df[
        ["person_id", "label", "node_type", "degree", "weighted_degree", "pagerank", "community"]
    ].head(20)
)

print("\nTOP 20 — graf skierowany według weighted_in_degree")
print(
    directed_metrics_df[
        ["person_id", "label", "node_type", "in_degree", "out_degree",
         "weighted_in_degree", "weighted_out_degree", "pagerank", "authority", "hub"]
    ].head(20)
)


#%% 12. Podsumowanie tekstowe

largest_cc_size = 0

if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
    largest_cc_size = len(max(nx.connected_components(G), key=len))

summary = f"""
Arkusz / PBL — sieć osób
========================

Graf współwystąpień:
- liczba węzłów: {G.number_of_nodes()}
- liczba krawędzi: {G.number_of_edges()}
- liczba składowych: {nx.number_connected_components(G) if G.number_of_nodes() else 0}
- największa składowa: {largest_cc_size}

Graf skierowany autor → osoba:
- liczba węzłów: {DG.number_of_nodes()}
- liczba krawędzi: {DG.number_of_edges()}

Pliki wyjściowe:
- arkusz_person_cooccurrence_sigma_style.graphml
- arkusz_person_cooccurrence_sigma_style.gexf
- arkusz_person_cooccurrence_network.html, jeśli ipysigma jest zainstalowana
- person_network_metrics_sigma_style.csv
- person_edges_cooccurrence_cleaned.csv
- arkusz_person_interpretive_directed_sigma_style.graphml
- arkusz_person_interpretive_directed_sigma_style.gexf
- arkusz_person_interpretive_directed_network.html, jeśli ipysigma jest zainstalowana
- person_directed_network_metrics_sigma_style.csv
- person_edges_directed_cleaned.csv
- arkusz_person_network_top_sigma_style.png
""".strip()

(OUT / "summary_sigma_style.txt").write_text(summary, encoding="utf-8")

print("\n" + summary)
print(f"\nWyniki zapisano w katalogu: {OUT.resolve()}")