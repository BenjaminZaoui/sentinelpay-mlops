# Données

- `creditcard.csv` : jeu historique étiqueté (*Credit Card Fraud Detection*), fourni par l'enseignant via « les fichiers de la classe ». À placer ici. Il n'est jamais commité dans Git (voir `.gitignore`) : il est versionné par DVC à partir de la mission M2.
- `prod/` : lots de production mensuels, produits par `python tools/make_prod_batches.py` (missions M6 à M8). Non commités.

Vérification rapide :

```python
import pandas as pd
df = pd.read_csv("data/creditcard.csv")
print(df.shape)                    # (284807, 31)
print(df["Class"].value_counts())  # 284315 légitimes, 492 fraudes
```
