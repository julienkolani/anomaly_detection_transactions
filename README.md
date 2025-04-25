# Projet : Détection d'anomalies dans les transactions bancaires avec K-means

## Objectif

Utiliser K-means pour identifier automatiquement les comportements inhabituels dans un ensemble de transactions synthétiques. Les clusters modélisent différents comportements et les anomalies peuvent être identifiées en observant les points trop éloignés des centres.

---

## Étapes du Projet

### Étape 1 : Chargement des données

**Objectif :** Charger le fichier `transactions.csv` contenant 20 000 transactions.

**À faire :**

- Lire le fichier CSV.
- Afficher les premières lignes pour inspection.

#### Signature de la fonction :

```python
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the transaction dataset from the given CSV file.

    :param file_path: Path to the CSV file
    :return: Pandas DataFrame with the transaction data
    """
    pass
```

---

### Étape 2 : Sélection des caractéristiques pertinentes

**Objectif :** Choisir les colonnes utiles pour la détection d’anomalies (exemple : `initial_amount`, `transfer_amount`, `amount_received`, `final_amount`).

**À faire :**

- Extraire uniquement les colonnes numériques pertinentes.
- Ne pas inclure les identifiants ou les dates.

#### Signature de la fonction :

```python
def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select relevant numerical features for clustering.

    :param df: Full transaction DataFrame
    :return: DataFrame with selected features only
    """
    pass
```

---

### Étape 3 : Préparation et normalisation des données

**Objectif :** Normaliser les données avant de passer à K-means.

**À faire :**

- Appliquer `StandardScaler`.
- Retourner les données normalisées.

#### Signature de la fonction :

```python
from sklearn.preprocessing import StandardScaler

def normalize_features(df: pd.DataFrame) -> np.ndarray:
    """
    Normalize the selected features using StandardScaler.

    :param df: DataFrame of selected features
    :return: Normalized NumPy array of features
    """
    pass
```

---

### Étape 4 : Clustering avec K-means

**Objectif :** Regrouper les transactions en 6 clusters.

**À faire :**

- Appliquer `KMeans` avec 6 clusters.
- Retourner le modèle entraîné et les labels.

#### Signature de la fonction :

```python
from sklearn.cluster import KMeans

def apply_kmeans(data: np.ndarray, n_clusters: int = 6) -> tuple:
    """
    Apply KMeans clustering to the normalized transaction data.

    :param data: Normalized feature array
    :param n_clusters: Number of clusters
    :return: Tuple (trained model, labels)
    """
    pass
```

---

### Étape 5 : Détection des anomalies

**Objectif :** Identifier les transactions anormales en fonction de leur distance au centroïde du cluster.

**À faire :**

- Calculer les distances de chaque point à son centroïde.
- Retourner les indices où la distance dépasse un seuil défini.

#### Signature de la fonction :

```python
def detect_anomalies(data: np.ndarray, model: KMeans, threshold: float) -> np.ndarray:
    """
    Detect transactions that are far from their cluster center.

    :param data: Normalized data
    :param model: Trained KMeans model
    :param threshold: Distance threshold
    :return: Array of indices of anomalous transactions
    """
    pass
```

---

### Étape 6 : Affichage des résultats

**Objectif :** Visualiser les clusters et mettre en évidence les anomalies détectées.

**À faire :**

- Utiliser un scatter plot avec 2 dimensions (réduction de dimensions si nécessaire).
- Mettre en rouge les anomalies.

#### Signature de la fonction :

```python
def plot_clusters(data: np.ndarray, labels: np.ndarray, anomalies: np.ndarray):
    """
    Plot the clustered transactions and highlight anomalies.

    :param data: Normalized data
    :param labels: Labels assigned by KMeans
    :param anomalies: Indices of anomalies
    """
    pass
```

---

### Étape 7 : Script principal

**Objectif :** Exécuter toutes les étapes ensemble.

**À faire :**

1. Lire les données.
2. Sélectionner les caractéristiques.
3. Normaliser les données.
4. Appliquer K-means.
5. Détecter les anomalies.
6. Visualiser les résultats.
7. Ajuster éventuellement le seuil.

---

### Étape 8 : Interprétation des résultats et rédaction d’un rapport

**Objectif :** Analyser les clusters et les anomalies détectées.

**À faire :**

- **Analyser les clusters :**

  - Examiner les moyennes des caractéristiques (`initial_amount`, `transfer_amount`, etc.).
  - Déduire le type de comportement représenté (normal, overdraft, ghost transaction, etc.).
- **Analyser les anomalies détectées :**

  - Combien de transactions sont considérées comme anomalies ?
  - Ces anomalies appartiennent-elles à un ou plusieurs clusters spécifiques ?
  - Que révèlent-elles (transactions fantômes, virements incohérents, etc.) ?
- **Écrire un rapport synthétique :**

  - Présenter les résultats (nombre de clusters, nombre d’anomalies).
  - Interpréter les patterns détectés (habitudes clients, anomalies critiques).
  - Proposer des recommandations si nécessaire (ex. surveillance accrue de certains types de transactions).

---
