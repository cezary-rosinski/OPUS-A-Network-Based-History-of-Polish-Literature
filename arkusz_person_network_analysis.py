from pathlib import Path
import itertools
import re
import json
import math
import zipfile
from collections import defaultdict, Counter

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# The script is now self-contained: it reads the input files bundled in the same folder.
SCRIPT_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
BASE = SCRIPT_DIR
OUT = SCRIPT_DIR
OUT.mkdir(exist_ok=True)

NODES_PATH = BASE / 'nodes.csv'
EDGES_PATH = BASE / 'edges.csv'

PERSON_LABELS = {'Author', 'Creator', 'Person', 'IndexPerson'}
ROLE_BY_REL = {
    'HAS_AUTHOR': 'author',
    'HAS_CREATOR': 'creator',
    'INDEXES_PERSON': 'indexed_person',
    'HAS_PERSON_ROLE': 'person_role',
}
ROLE_PRIORITY = {'author': 1, 'creator': 2, 'indexed_person': 3, 'person_role': 4}

nodes = pd.read_csv(NODES_PATH)
edges = pd.read_csv(EDGES_PATH)

# ---------- helpers ----------
def norm_name(name: str) -> str:
    if pd.isna(name):
        return ''
    s = str(name).strip()
    s = re.sub(r'\s+', ' ', s)
    # name is used as stable person key only when no universal person ID exists across role tables
    return s.casefold()

def clean_year(y):
    try:
        if pd.isna(y):
            return ''
        return int(float(y))
    except Exception:
        return ''

def graphml_value(v):
    if v is None:
        return ''
    try:
        if pd.isna(v):
            return ''
    except Exception:
        pass
    if isinstance(v, (str, int, float, bool)):
        return v
    return str(v)

def pair_key(a, b):
    return tuple(sorted((a, b)))

# ---------- person nodes ----------
person_rows = nodes[nodes['label'].isin(PERSON_LABELS)].copy()
person_rows['canonical_name'] = person_rows['name'].fillna('').astype(str).str.strip()
person_rows['person_key'] = person_rows['canonical_name'].map(norm_name)
person_rows = person_rows[person_rows['person_key'] != ''].copy()
person_rows['person_node_id'] = 'PersonName:' + person_rows['person_key']

kg_to_person = dict(zip(person_rows['kg_id'], person_rows['person_node_id']))
kg_to_role_label = dict(zip(person_rows['kg_id'], person_rows['label']))
kg_to_name = dict(zip(person_rows['kg_id'], person_rows['canonical_name']))

# Aggregate role-table nodes that have the same name into one person node.
agg_nodes = []
for pid, g in person_rows.groupby('person_node_id', dropna=False):
    names = [x for x in g['canonical_name'].dropna().astype(str).unique() if x.strip()]
    display_name = Counter(names).most_common(1)[0][0] if names else pid.replace('PersonName:', '')
    labels = sorted(g['label'].dropna().unique())
    ids_by_label = {}
    for lab in labels:
        ids_by_label[lab] = sorted(g.loc[g['label'] == lab, 'kg_id'].dropna().astype(str).unique().tolist())
    agg_nodes.append({
        'person_id': pid,
        'label': display_name,
        'name': display_name,
        'source_labels': '|'.join(labels),
        'role_table_ids': json.dumps(ids_by_label, ensure_ascii=False),
        'source_node_count': int(len(g)),
    })
person_nodes = pd.DataFrame(agg_nodes)

# ---------- record metadata ----------
record_rows = nodes[nodes['label'] == 'Record'].copy()
record_meta = {}
for _, r in record_rows.iterrows():
    record_meta[r['kg_id']] = {
        'zapis_id': r.get('zapis_id'),
        'title': r.get('title'),
        'year': clean_year(r.get('year')),
        'source_id': r.get('source_id'),
        'source_no': r.get('source_no'),
        'pages': r.get('pages'),
        'record_type': r.get('record_type'),
        'in_seed': r.get('in_seed'),
    }

# ---------- record-person incidence ----------
inc_rows = []
for _, e in edges.iterrows():
    rel = e['relationship']
    if rel not in ROLE_BY_REL:
        continue
    start = e['start_id']
    end = e['end_id']
    if start not in record_meta or end not in kg_to_person:
        continue
    meta = record_meta[start]
    inc_rows.append({
        'record_id': start,
        'zapis_id': meta['zapis_id'],
        'record_title': meta['title'],
        'year': meta['year'],
        'source_no': meta['source_no'],
        'record_type': meta['record_type'],
        'person_id': kg_to_person[end],
        'person_name': kg_to_name[end],
        'source_role_label': kg_to_role_label[end],
        'role': ROLE_BY_REL[rel],
        'relationship': rel,
    })
incidence = pd.DataFrame(inc_rows).drop_duplicates()
incidence.to_csv(OUT / 'record_person_incidence.csv', index=False)

# Enrich person node attributes with roles and record counts.
role_counts = incidence.groupby(['person_id', 'role']).size().unstack(fill_value=0)
record_counts = incidence.groupby('person_id')['record_id'].nunique().rename('record_count')
year_min = incidence.dropna(subset=['year']).groupby('person_id')['year'].min().rename('first_year')
year_max = incidence.dropna(subset=['year']).groupby('person_id')['year'].max().rename('last_year')
person_nodes = person_nodes.set_index('person_id').join(role_counts, how='left').join(record_counts, how='left').join(year_min, how='left').join(year_max, how='left').fillna({
    'author': 0, 'creator': 0, 'indexed_person': 0, 'person_role': 0, 'record_count': 0
}).reset_index()
for col in ['author','creator','indexed_person','person_role','record_count']:
    if col in person_nodes:
        person_nodes[col] = person_nodes[col].astype(int)

# ---------- undirected person-person co-occurrence network ----------
# Edge = two people co-occur in the same bibliographic record. We keep role-pair evidence.
edge_acc = {}
record_groups = incidence.groupby('record_id')
for record_id, g in record_groups:
    meta = record_meta.get(record_id, {})
    # one row per person-role in record; use distinct person+role to retain evidence
    rows = g[['person_id','person_name','role']].drop_duplicates().to_dict('records')
    # avoid self-pairs after name collapsing
    by_person = defaultdict(set)
    names_by_person = {}
    for row in rows:
        by_person[row['person_id']].add(row['role'])
        names_by_person[row['person_id']] = row['person_name']
    persons = sorted(by_person)
    for a, b in itertools.combinations(persons, 2):
        key = pair_key(a, b)
        if key not in edge_acc:
            edge_acc[key] = {
                'source': key[0], 'target': key[1], 'weight': 0,
                'record_ids': set(), 'zapis_ids': set(), 'years': set(), 'role_pairs': set(),
                'examples': []
            }
        acc = edge_acc[key]
        acc['weight'] += 1
        acc['record_ids'].add(record_id)
        zapis = meta.get('zapis_id')
        if not pd.isna(zapis):
            try: acc['zapis_ids'].add(str(int(float(zapis))))
            except Exception: acc['zapis_ids'].add(str(zapis))
        if meta.get('year') is not None:
            acc['years'].add(str(meta.get('year')))
        for ra in by_person[a]:
            for rb in by_person[b]:
                # canonical role-pair for undirected edge
                pair = tuple(sorted((ra, rb), key=lambda x: ROLE_PRIORITY.get(x, 99)))
                acc['role_pairs'].add(f'{pair[0]}--{pair[1]}')
        if len(acc['examples']) < 5:
            title = meta.get('title')
            if isinstance(title, str) and title.strip():
                acc['examples'].append(title.strip()[:240])

und_edges = []
for (a, b), acc in edge_acc.items():
    und_edges.append({
        'source': a, 'target': b, 'weight': acc['weight'],
        'record_count': len(acc['record_ids']),
        'zapis_ids': '|'.join(sorted(acc['zapis_ids'], key=lambda x: (len(x), x))),
        'years': '|'.join(sorted(acc['years'])),
        'first_year': min([int(y) for y in acc['years']]) if acc['years'] else None,
        'last_year': max([int(y) for y in acc['years']]) if acc['years'] else None,
        'role_pairs': '|'.join(sorted(acc['role_pairs'])),
        'example_titles': ' || '.join(acc['examples']),
    })
person_edges = pd.DataFrame(und_edges)

# ---------- directed interpretive network ----------
# Edge = authorial agent points to the person construed as object/topic of record.
# This is not a citation edge; it is a bibliographic/documentary relation.
dir_acc = {}
for record_id, g in record_groups:
    meta = record_meta.get(record_id, {})
    authors = sorted(g.loc[g['role']=='author','person_id'].dropna().unique())
    target_roles = ['creator','indexed_person','person_role']
    targets = sorted(g.loc[g['role'].isin(target_roles),'person_id'].dropna().unique())
    for a in authors:
        for t in targets:
            if a == t:
                continue
            key = (a, t)
            if key not in dir_acc:
                dir_acc[key] = {'source': a, 'target': t, 'weight': 0, 'zapis_ids': set(), 'years': set(), 'target_roles': set(), 'examples': []}
            acc = dir_acc[key]
            acc['weight'] += 1
            zapis = meta.get('zapis_id')
            if not pd.isna(zapis):
                try: acc['zapis_ids'].add(str(int(float(zapis))))
                except Exception: acc['zapis_ids'].add(str(zapis))
            if meta.get('year') is not None:
                acc['years'].add(str(meta.get('year')))
            acc['target_roles'].update(g.loc[g['person_id']==t,'role'].dropna().unique().tolist())
            if len(acc['examples']) < 5:
                title = meta.get('title')
                if isinstance(title, str) and title.strip():
                    acc['examples'].append(title.strip()[:240])

dir_edges = []
for (a, t), acc in dir_acc.items():
    dir_edges.append({
        'source': a, 'target': t, 'weight': acc['weight'],
        'zapis_ids': '|'.join(sorted(acc['zapis_ids'], key=lambda x: (len(x), x))),
        'years': '|'.join(sorted(acc['years'])),
        'first_year': min([int(y) for y in acc['years']]) if acc['years'] else None,
        'last_year': max([int(y) for y in acc['years']]) if acc['years'] else None,
        'relationship': 'AUTHORS_RECORD_ABOUT_PERSON',
        'target_roles': '|'.join(sorted(acc['target_roles'])),
        'example_titles': ' || '.join(acc['examples']),
    })
interpretive_edges = pd.DataFrame(dir_edges)

# ---------- build graphs and metrics ----------
G = nx.Graph()
for _, r in person_nodes.iterrows():
    G.add_node(r['person_id'], **{k: graphml_value(v) for k, v in r.to_dict().items() if k != 'person_id'})
for _, e in person_edges.iterrows():
    G.add_edge(e['source'], e['target'], **{k: graphml_value(v) for k, v in e.to_dict().items() if k not in {'source','target'}})

DG = nx.DiGraph()
for _, r in person_nodes.iterrows():
    DG.add_node(r['person_id'], **{k: graphml_value(v) for k, v in r.to_dict().items() if k != 'person_id'})
for _, e in interpretive_edges.iterrows():
    if DG.has_edge(e['source'], e['target']):
        DG[e['source']][e['target']]['weight'] += int(e.get('weight', 1))
    else:
        DG.add_edge(e['source'], e['target'], **{k: graphml_value(v) for k, v in e.to_dict().items() if k not in {'source','target'}})

metrics = pd.DataFrame({'person_id': list(G.nodes())})
metrics['name'] = metrics['person_id'].map(nx.get_node_attributes(G,'name'))
metrics['degree'] = metrics['person_id'].map(dict(G.degree())).fillna(0).astype(int)
metrics['weighted_degree'] = metrics['person_id'].map(dict(G.degree(weight='weight'))).fillna(0).astype(int)
metrics['record_count'] = metrics['person_id'].map(nx.get_node_attributes(G,'record_count')).fillna(0).astype(int)
metrics['author_count'] = metrics['person_id'].map(nx.get_node_attributes(G,'author')).fillna(0).astype(int)
metrics['creator_count'] = metrics['person_id'].map(nx.get_node_attributes(G,'creator')).fillna(0).astype(int)
metrics['indexed_person_count'] = metrics['person_id'].map(nx.get_node_attributes(G,'indexed_person')).fillna(0).astype(int)

# Largest connected component metrics for stable centralities
if G.number_of_nodes() and G.number_of_edges():
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    lcc_nodes = comps[0]
    GL = G.subgraph(lcc_nodes).copy()
    bt = nx.betweenness_centrality(GL, k=min(200, GL.number_of_nodes()), seed=42, normalized=True)
    try:
        ev = nx.eigenvector_centrality(GL, weight='weight', max_iter=300)
    except Exception:
        ev = {n: 0.0 for n in GL.nodes()}
    core = nx.core_number(GL)
    comms = list(nx.community.greedy_modularity_communities(GL, weight='weight'))
    comm_map = {}
    for i, c in enumerate(comms):
        for n in c:
            comm_map[n] = i
else:
    bt = ev = core = comm_map = {}
    comms = []
metrics['betweenness_lcc'] = metrics['person_id'].map(bt).fillna(0.0)
metrics['eigenvector_lcc'] = metrics['person_id'].map(ev).fillna(0.0)
metrics['core_number_lcc'] = metrics['person_id'].map(core).fillna(0).astype(int)
metrics['community_lcc'] = metrics['person_id'].map(comm_map)

if DG.number_of_nodes() and DG.number_of_edges():
    metrics['in_degree_interpretive'] = metrics['person_id'].map(dict(DG.in_degree())).fillna(0).astype(int)
    metrics['out_degree_interpretive'] = metrics['person_id'].map(dict(DG.out_degree())).fillna(0).astype(int)
    metrics['weighted_in_degree_interpretive'] = metrics['person_id'].map(dict(DG.in_degree(weight='weight'))).fillna(0).astype(int)
    metrics['weighted_out_degree_interpretive'] = metrics['person_id'].map(dict(DG.out_degree(weight='weight'))).fillna(0).astype(int)
    try:
        pr = nx.pagerank(DG, weight='weight', alpha=0.85)
    except Exception:
        pr = {n: 0.0 for n in DG.nodes()}
    metrics['pagerank_interpretive'] = metrics['person_id'].map(pr).fillna(0.0)
else:
    for col in ['in_degree_interpretive','out_degree_interpretive','weighted_in_degree_interpretive','weighted_out_degree_interpretive']:
        metrics[col] = 0
    metrics['pagerank_interpretive'] = 0.0

# Attach metrics to graph nodes
for _, r in metrics.iterrows():
    attrs = {k: graphml_value(v) for k, v in r.to_dict().items() if k not in {'person_id','name'}}
    if G.has_node(r['person_id']):
        G.nodes[r['person_id']].update(attrs)
    if DG.has_node(r['person_id']):
        DG.nodes[r['person_id']].update(attrs)

# Communities summary
community_rows = []
for i, c in enumerate(comms):
    sub = metrics[metrics['person_id'].isin(c)].sort_values(['weighted_degree','degree'], ascending=False)
    community_rows.append({
        'community': i,
        'size': len(c),
        'weighted_degree_sum': int(sub['weighted_degree'].sum()),
        'top_people': ' | '.join(sub['name'].head(12).fillna('').tolist()),
    })
communities = pd.DataFrame(community_rows).sort_values(['size','weighted_degree_sum'], ascending=False) if community_rows else pd.DataFrame()

# Save outputs
person_nodes.to_csv(OUT / 'person_nodes.csv', index=False)
person_edges.sort_values('weight', ascending=False).to_csv(OUT / 'person_edges_cooccurrence.csv', index=False)
interpretive_edges.sort_values('weight', ascending=False).to_csv(OUT / 'person_edges_interpretive_directed.csv', index=False)
metrics.sort_values(['weighted_degree','degree'], ascending=False).to_csv(OUT / 'person_network_metrics.csv', index=False)
communities.to_csv(OUT / 'person_communities.csv', index=False)

nx.write_graphml(G, OUT / 'arkusz_person_cooccurrence_network.graphml')
nx.write_gexf(G, OUT / 'arkusz_person_cooccurrence_network.gexf')
nx.write_graphml(DG, OUT / 'arkusz_person_interpretive_directed_network.graphml')
nx.write_gexf(DG, OUT / 'arkusz_person_interpretive_directed_network.gexf')

# Static visualizations
plt.figure(figsize=(11, 7))
top = metrics.sort_values('weighted_degree', ascending=False).head(25).sort_values('weighted_degree')
plt.barh(top['name'], top['weighted_degree'])
plt.xlabel('Weighted degree')
plt.title('Arkusz: najważniejsze osoby w sieci współwystąpień')
plt.tight_layout()
plt.savefig(OUT / 'top_people_weighted_degree.png', dpi=200)
plt.close()

plt.figure(figsize=(13, 10))
# subgraph of top nodes by weighted degree, plus edges among them
sub_nodes = set(metrics.sort_values('weighted_degree', ascending=False).head(120)['person_id'])
SG = G.subgraph(sub_nodes).copy()
pos = nx.spring_layout(SG, k=0.45, seed=42, weight='weight')
node_sizes = [40 + 8 * SG.nodes[n].get('weighted_degree', 1) for n in SG.nodes()]
edge_widths = [0.2 + 0.25 * math.log1p(SG[u][v].get('weight', 1)) for u, v in SG.edges()]
nx.draw_networkx_edges(SG, pos, width=edge_widths, alpha=0.25)
nx.draw_networkx_nodes(SG, pos, node_size=node_sizes, alpha=0.75)
label_nodes = set(metrics.sort_values('weighted_degree', ascending=False).head(25)['person_id'])
labels = {n: SG.nodes[n].get('name', n) for n in SG.nodes() if n in label_nodes}
nx.draw_networkx_labels(SG, pos, labels=labels, font_size=7)
plt.title('Arkusz: osobowa sieć współwystąpień (TOP 120 wg weighted degree)')
plt.axis('off')
plt.tight_layout()
plt.savefig(OUT / 'person_network_top120.png', dpi=250)
plt.close()

# Report
summary = {
    'people': G.number_of_nodes(),
    'cooccurrence_edges': G.number_of_edges(),
    'interpretive_directed_edges': DG.number_of_edges(),
    'connected_components': nx.number_connected_components(G) if G.number_of_nodes() else 0,
    'largest_component_size': len(max(nx.connected_components(G), key=len)) if G.number_of_nodes() else 0,
    'communities_lcc': len(comms),
}
report = []
report.append('# Arkusz — osobowa sieć relacji z danych PBL\n')
report.append('## Model\n')
report.append('Sieć zawiera wyłącznie osoby. Węzły ról osobowych z grafu wiedzy (`Author`, `Creator`, `Person`, `IndexPerson`) zostały scalone po znormalizowanej nazwie osoby. Krawędzie powstały przez projekcję rekordów bibliograficznych: jeżeli dwie osoby występują przy tym samym zapisie PBL, otrzymują relację współwystąpienia. Dodatkowo wygenerowano skierowaną sieć interpretacyjną `author -> person`, w której autor zapisu jest łączony z osobą będącą przedmiotem/opisywanym twórcą/osobą indeksowaną.\n')
report.append('## Liczby\n')
for k, v in summary.items():
    report.append(f'- {k}: {v}')
report.append('\n## Najważniejsze osoby wg weighted degree\n')
for _, r in metrics.sort_values('weighted_degree', ascending=False).head(20).iterrows():
    report.append(f"- {r['name']}: weighted_degree={int(r['weighted_degree'])}, degree={int(r['degree'])}, records={int(r['record_count'])}, community={r['community_lcc']}")
report.append('\n## Interpretacja metodologiczna\n')
report.append('Ta sieć nie pokazuje relacji biograficznych sensu stricto. Pokazuje relacje wytworzone przez praktykę dokumentacyjną PBL: współobecność osób w tym samym zapisie bibliograficznym oraz relację autor-opisywana osoba. To dobra warstwa dla narracji o bibliografii jako aparacie badawczym: rekord bibliograficzny nie jest końcowym opisem jednostkowym, ale miejscem, w którym można rekonstruować pole aktorów, tematów i mediacji krytycznoliterackich wokół czasopisma.')
(OUT / 'person_network_report.md').write_text('\n'.join(report), encoding='utf-8')

# Copy/reusable script into output with more readable name
(Path(__file__).read_text(encoding='utf-8'))

# Zip
zip_path = Path('/mnt/data/arkusz_person_network_package.zip')
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for p in OUT.rglob('*'):
        zf.write(p, p.relative_to(OUT.parent))

print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f'Wrote {zip_path}')
