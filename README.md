# Backtesting Framework

Framework de backtesting quantitatif basé sur les rendements, avec extraction Bloomberg, interface Streamlit et espace de recherche séparé pour la réplication du papier de pairs trading par ondelettes.

## Vue d'ensemble

Le repo est maintenant organisé autour de deux périmètres distincts :

- `config.py` : configuration globale du backtester générique
- `research/config.py` : configuration locale de la réplication du papier

Le backtester standard et la réplication du papier partagent les loaders et les données, mais gardent des paramètres et des sorties séparés.

## Structure

```
Backtesting_Framework/
├── app/                              # Application Streamlit
│   ├── main.py                       # Entry point : py -m streamlit run app/main.py
│   ├── sidebar.py                    # Navigation latérale
│   ├── registry.py                   # Registre stratégies / allocations
│   ├── data.py                       # Loaders cachés pour l'app
│   ├── styles.py                     # CSS
│   └── steps/                        # Wizard data -> backtest -> réplication papier
├── allocation/                       # Méthodes d'allocation
├── extraction/                       # Extraction Bloomberg + API wrapper
│   ├── bloomberg_api.py
│   ├── bbg_members.py
│   ├── bbg_returns.py
│   └── bbg_riskfree.py
├── indicators/                       # Indicateurs perf / risque
├── loaders/                          # Chargement Polars / Bloomberg / parquet
├── optimization/                     # Grid search walk-forward
├── portfolio/                        # Moteur de backtest
├── signals/                          # Stratégies du framework
│   ├── moving_average.py
│   ├── momentum.py
│   └── pairs_trading.py
├── visualisation/                    # Figures du backtester
├── research/                         # Recherche et réplication du papier
│   ├── config.py                     # Paramètres fixes de la réplication papier
│   ├── main.py                       # CLI principale de la réplication papier
│   ├── docs/                         # Documentation / papier source
│   ├── outputs/                      # Tables et figures générées
│   └── paper_replication/            # Code de la réplication du papier
├── data/                             # Données parquet partagées
├── config.py                         # Configuration globale du backtester
├── requirements.txt
└── README.md
```

## Installation

Sur Windows :

```bash
py -m pip install -r requirements.txt
```

Bloomberg API optionnelle :

```bash
py -m pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi
```

## Lancement

Application Streamlit :

```bash
py -m streamlit run app/main.py
```

Smoke test de la réplication papier :

```bash
py research/main.py --index-id SPX --max-periods 1 --no-write-outputs --quiet
```

## Tests et CI

Lancement local de la suite :

```bash
py -m unittest discover -s tests
```

CI GitHub Actions :

- workflow dans [.github/workflows/tests.yml](.github/workflows/tests.yml)
- exécution sur `push` et `pull_request` vers `main`
- vérification de la suite `unittest` et du démarrage de l'aide CLI `research/main.py --help`

## Workflow Streamlit

L'application suit un workflow en 7 étapes :

1. `Data Status` : vérification des jeux de données et extraction Bloomberg conditionnelle
2. `Index Selection` : sélection de l'univers
3. `Strategy & Alloc` : choix stratégie / allocation
4. `Parameters` : réglage des paramètres et grid search éventuel
5. `Execution` : exécution du backtest
6. `Results` : métriques et visualisations
7. `Paper Replication` : exécution séparée de la réplication du papier

## Configurations

Configuration globale du backtester dans [config.py](config.py) :

- chemins de données partagées
- univers d'indices et devises
- taux sans risque
- fréquences de rebalancement
- grilles de paramètres du backtester standard

Configuration locale de recherche dans [research/config.py](research/config.py) :

- chemins `research/docs` et `research/outputs`
- paramètres fixes de la réplication du papier
- options de sortie tables / figures

## Données et artefacts

Les fichiers de [data](data) et le PDF de [research/docs](research/docs) restent versionnés pour garder un repo reproductible sans étape de bootstrap externe. En revanche, les artefacts générés pendant les runs ne doivent pas être commités :

- [research/outputs](research/outputs) est réservé aux sorties générées
- `research/outputs/tables` et `research/outputs/figures` sont ignorés par git
- les données peuvent être régénérées via l'étape `Data Status` de l'app ou via les scripts d'extraction Bloomberg

## Stratégies disponibles

- `Moving Average Crossover`
- `Momentum`
- `Pairs Trading (Wavelet)` dans le framework

La réplication papier complète reste séparée dans [research/paper_replication](research/paper_replication) pour éviter de mélanger logique de recherche et moteur de backtest générique.

## Principes du backtester

- calcul de P&L basé sur les rendements journaliers
- dérive des poids entre deux rebalancements
- application du rendement du jour avant rebalancement
- membership point-in-time pour limiter le survivorship bias
- taux sans risque journaliers réels par devise
- signaux calculés en expanding window
