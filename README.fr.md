# tradelab

*[English version](README.md)*

Banc d'essai de stratégies de trading : simulation réaliste d'un compte à marge, puis **verdict statistique** sur une période que la stratégie n'a pas vue.

Le projet part d'un cas réel : un bot de trading « grille + short + levier adaptatif » qui tournait avec de l'argent réel sur Kraken. Sa logique est ici portée en fonction pure, rejouée sur six ans de données ETH/USDT de Binance (non incluses dans ce dépôt, voir [Données](#données)) et soumise à un juge qui demande : *ce résultat se distingue-t-il du hasard et de la simple détention ?*

> **Résultat : la stratégie n'est démontrée rentable dans aucune des deux variantes testées.** Avec les frais réels du bot, elle perd de l'argent sur la période de validation (−90 % avec le levier du bot, −73 % sans levier), alors que garder l'ETH aurait rapporté +108 %. Voir [les résultats](#résultats). C'est le rôle de l'outil : mettre à l'épreuve une idée qui a l'air bonne avant d'y mettre de l'argent.

## Ce que ce projet démontre

| Compétence | Où la voir |
|---|---|
| **Logique de décision séparée des entrées/sorties** : la stratégie est une fonction pure, le même code sert au backtest et pourrait servir derrière un courtier | `src/tradelab/strategy.py` |
| **Simulation de compte à marge** : levier, marge bloquée, rollover sur le montant emprunté, liquidation au pire prix de la bougie | `src/tradelab/account.py`, `costs.py` |
| **Rigueur statistique** : fenêtre de validation à capital neuf, bootstrap par blocs de mois, placebo par décalage circulaire, correction de Bonferroni | `src/tradelab/validation.py` |
| **Ingénierie de données** : lecture sécurisée de `.npz` (pas de pickle), rééchantillonnage, rapport de qualité | `src/tradelab/data.py` |
| **Configurable sans toucher au code** : paramètres de la stratégie (`--set`, `--params`), niveaux de levier à comparer (`--leverage`), analyses à lancer (`--analyses`), décompte honnête des hypothèses (`--tests-counted`) | `src/tradelab/options.py`, `examples/` |
| **Robustesse opérationnelle** : état écrit de façon atomique, sauvegarde `.bak`, refus de repartir à vide | `src/tradelab/state.py` |
| **Qualité** : 285 tests, dont des tests écrits à partir de défauts réels du bot d'origine ; mutations volontaires : 10 sur le code d'origine, 42 sur 43 détectées sur le code des options (la 43ᵉ a révélé un contrôle redondant, retiré depuis) | `tests/` |

## Démarrage

Python 3.10 ou plus récent. Aucune clé, aucun compte, aucun accès réseau.

```bash
pip install -e ".[dev]"
pytest                                  # 285 tests, ~40 s
tradelab backtest --sensitivity         # ~15 s, sur des données FICTIVES (voir ci-dessous)
tradelab carry                          # portage du financement, sur des données FICTIVES
tradelab params                         # ce qu'on peut changer : paramètres, niveaux de levier, analyses
```

Par défaut, les commandes lisent `data/synthetic/`, des données **inventées**. Leurs chiffres diffèrent donc de ceux de la section [Résultats](#résultats), qui viennent de données réelles. Le rapport affiche « données FICTIVES » en tête quand c'est le cas.

## Choisir ce qu'on teste

Le lancement par défaut reproduit le bot d'origine (levier adaptatif, et sans levier). Sans toucher au code, tu peux changer les paramètres de la stratégie, choisir les niveaux de levier à comparer et choisir les analyses à lancer. `tradelab params` affiche tout, avec les valeurs par défaut et les plages acceptées.

```bash
tradelab params                                                   # paramètres, plages, niveaux de levier, analyses, format du fichier
tradelab backtest --set spread=0.02 --set levels=8                # change des paramètres (virgule décimale acceptée : spread=0,02)
tradelab backtest --params examples/params.example.json           # la même chose depuis un fichier JSON ; --set passe après le fichier
tradelab backtest --leverage 1,2,3                                # une hypothèse par niveau (adaptive = le ×2/×3/×5 du bot)
tradelab backtest --analyses sensitivity --set levels=6           # seulement le tableau de sensibilité aux coûts
tradelab backtest --tests-counted 20                              # déclare 20 tests au total pour la correction de Bonferroni
```

- **13 paramètres réglables** : niveaux et écart de la grille, intervalle de reconstruction, seuils ADX et RSI, stop et objectif du short, quantités minimales. Un nom inconnu, une valeur hors plage, un mauvais type ou un couple incohérent (`rsi_cover` doit rester inférieur à `rsi_short`) est refusé avec un message, **avant** toute lecture de données. Le levier se choisit avec `--leverage`, pas avec les paramètres.
- **Chaque hypothèse lancée compte pour la correction de Bonferroni** : trois niveaux de levier, c'est α/3. Si tu as essayé d'autres configurations ailleurs, déclare le total avec `--tests-counted` ; il peut relever le compte, jamais le descendre sous ce que ce lancement exécute.
- **Changer les paramètres change le sens du verdict.** Le juge suppose que la stratégie a été figée avant de regarder VAL. Ajuster les paramètres jusqu'à ce que VAL soit bon rend le verdict sans valeur, correction ou pas ; le rapport le rappelle chaque fois que des paramètres sont modifiés. Règle sur DEV, ou fixe les valeurs d'abord.
- **Figé volontairement** : les règles de décision (`strategy.decide`), la validation minimale de 24 mois, le bootstrap par blocs de mois et le placebo. La stratégie est paramétrable, pas remplaçable : tester une autre stratégie, c'est écrire un autre `decide()`.
- Sans option, le rapport et le JSON sont identiques, caractère pour caractère, à ceux de la version précédente (comparés à des sorties enregistrées).

## Données

**Ce dépôt ne contient aucune donnée de marché réelle.** Les fichiers de `data/synthetic/` (1,2 Mo) sont fictifs : une marche aléatoire à volatilité variable et à queues épaisses (un « ETH » en bougies d'une heure de janvier 2020 à octobre 2026, sans trou) et un financement quotidien inventé. `src/tradelab/synthetic.py` les produit avec une graine fixe, `python scripts/make_synthetic.py` les régénère à l'identique, et un test vérifie que les fichiers du dépôt correspondent au générateur. Ils servent à lancer l'outil et les tests sans rien télécharger ; ils ne disent rien d'un vrai marché.

Les [résultats](#résultats) ci-dessous, eux, ont été mesurés par l'auteur sur des données réelles de Binance : ETH/USDT en bougies d'une heure et financement quotidien du BTC. Ces données ne sont pas redistribuées ici : les [conditions d'utilisation de Binance Vision](https://data.binance.vision/Binance_Vision-Terms_of_Use.pdf) les placent sous licence CC BY-NC-SA 4.0 (usage non commercial, attribution) et interdisent d'héberger les jeux de données. Pour refaire la mesure :

1. Télécharger les données auprès de Binance Vision, dans le respect de ces conditions.
2. Les convertir en CSV en UTC : `time,open,high,low,close,volume` pour les bougies d'une heure (fichier `.csv` ou `.csv.gz`), `date,funding_sum` pour la somme quotidienne du financement.
3. Lancer `tradelab backtest --data mes_bougies.csv` et `tradelab carry --funding mon_financement.csv`.

`scripts/build_sample.py` fait cette conversion à partir des archives `.npz` de l'auteur (format décrit dans `src/tradelab/data.py`) ; il écrit dans `data/real/`, dossier ignoré par git.

> Données de marché : Binance, via Binance Vision (CC BY-NC-SA 4.0). Les chiffres de la section Résultats ont été calculés par l'auteur, sans lien avec Binance, et sont montrés à titre d'illustration non commerciale.

## Résultats

*Mesures faites sur les données réelles décrites dans [Données](#données) ; elles ne se reproduisent pas avec `data/synthetic/`.*

Stratégie du bot d'origine, **paramètres inchangés** (rien n'est réglé ici), ETH/USDT en 1 h, départ à 50 de capital. Chaque fenêtre démarre avec un compte neuf, pour qu'une ruine survenue avant ne décide pas du verdict de la suivante. DEV (2020-2022) donne le contexte ; **seule VAL (2023-2026) juge**.

```
▌ Bot tel quel (levier adaptatif ×2/×3/×5)
  Fenêtre   Mois   Stratégie   Détention   Expo moy.   Moy. mensuelle   Borne basse   Excès (borne)   p placebo   Liq.
  DEV         36       -97 %       849 %        87 %          -2,91 %      -12,99 %        -16,93 %       0,918      2
  VAL         46       -90 %       108 %        83 %          -3,69 %       -8,95 %         -8,30 %       0,880      0
  VERDICT : non démontré

▌ Même stratégie sans levier (×1)
  DEV         36        37 %       849 %        36 %           1,30 %       -1,76 %         -3,43 %       0,705      0
  VAL         46       -73 %       108 %        51 %          -2,38 %       -5,74 %         -5,02 %       1,000      0
  VERDICT : non démontré

▌ Sensibilité aux coûts (période complète, sans levier, capital neuf)
  Scénario                    Rendement   Trades   Frais payés
  aucun coût                      184 %     2108           0,0
  frais 0,16 % + glissement        94 %     2150          64,2
  frais 0,26 % + glissement         6 %     2140          85,9
  frais 0,40 % + glissement       -77 %     2073         111,8
```

**Comment lire ces lignes.**

- Un verdict « démontré » exige trois conditions à la fois sur VAL : la moyenne mensuelle nette a une borne basse > 0 ; l'excès sur la détention de **même exposition moyenne** a une borne basse > 0 ; le test de placebo est significatif. Aucune n'est remplie, dans aucune variante.
- **Le levier aggrave tout.** Avec le levier du bot, deux liquidations (19 mai 2021 et 14 juin 2022) puis l'usure par les frais ruinent le compte simulé (−97 % sur DEV).
- **Le résultat dépend fortement des frais.** Sans aucun coût, la stratégie gagne, mais bien moins que la détention. À 0,16 % de frais elle est encore positive sur la période complète (mais déjà à −30 % sur la seule fenêtre VAL) ; à 0,26 % (la valeur du bot) elle ne fait que +6 % en 6,8 ans pour une ETH multipliée par 19 ; à 0,40 % elle perd 77 %. Dans la fenêtre VAL sans levier, les ordres sont minuscules (environ 6 de notionnel en moyenne) et leur gain moyen avant frais, glissement compris, est de l'ordre de 0,02 % du notionnel : les frais le dépassent dès qu'ils atteignent quelques dixièmes de pour cent.
- **p placebo ≥ 0,7 partout** : le profil d'exposition de la stratégie, décalé au hasard dans le temps, ferait aussi bien ou mieux. Son rythme d'entrée et de sortie n'apporte rien de mesurable.

### Portage du financement (l'autre piste du projet d'origine)

Acheter au comptant et vendre un perpétuel de même taille : le prix s'annule, il reste le financement versé entre acheteurs et vendeurs. Mesure sur le BTC, frictions de 2 %/an retirées d'office :

```
  Année   Jours   Brut /an   Net /an   Jours négatifs
  2020      366     17,2 %    15,2 %             14 %
  2021      365     30,6 %    28,6 %              8 %
  2022      365      4,2 %     2,2 %             20 %
  2023      365      7,9 %     5,9 %              8 %
  2024      366     11,9 %     9,9 %              7 %
  2025      365      5,1 %     3,1 %              8 %
  2026      273      2,9 %     0,9 %             24 %

  12 derniers mois   brut 3,4 %/an · net 1,4 %/an
  Rapporté au capital immobilisé (spot + marge du perpétuel) : 0,7 à 1,0 %/an
```

Le financement était positif la plupart des jours (de 76 % à 93 % selon l'année), ce qui va dans le sens du verdict « régulier » de l'étude d'origine. Mais l'essentiel venait de 2020-2021 : sur les 12 derniers mois il rapporte environ 3 % brut, **moins de 1,5 % net, et moins de 1 % du capital réellement immobilisé**. Hors mesure : liquidation de la jambe vendue en cas de hausse brutale, risque de plateforme, écart perpétuel/comptant, impôts, accès au produit depuis son pays, et le fait que le financement de Kraken n'est pas celui de Binance.

## Défauts trouvés dans le `bot.py` d'origine

Relevés en portant la logique (le code d'origine n'est pas inclus dans ce dépôt) :

1. **PnL affiché gonflé par le levier et sous-évalué en frais** (lignes 550, 622, 721, 768). Les frais sont retirés comme `× (1 − 2·FEES)`, c'est-à-dire une fraction du *profit* et non du notionnel, puis le résultat est multiplié par le levier alors que la quantité tient déjà compte du levier. Conséquence : un trade qui perd de l'argent peut s'afficher gagnant (`tests/test_costs.py`). Ce PnL alimente `eur_balance` (lignes 726 et 774). Les bots XRP et AVAX portent un commentaire « bug corrigé : *LEVERAGE en trop » ; le bot ETH et le bot SOL gardent le motif.
2. **Inventaire périmé pendant le cycle.** `eth_total` est lu une fois, puis plusieurs paliers de vente peuvent se déclencher dans la même boucle sans qu'il soit décrémenté : le bot peut vendre plus qu'il ne détient. Corrigé dans le port (`tests/test_strategy.py`, test de régression).
3. **`connect()` renvoie 2 valeurs en cas d'erreur, 4 attendues** (lignes 90 et 403) : une erreur de connexion fait planter le démarrage au lieu de s'arrêter proprement. Les bots SOL, XRP et AVAX renvoient bien 4 valeurs.
4. **Positions longue et courte supposées indépendantes.** Les scripts de réparation d'origine (`close_leftover_xrp.py`, `fix_after_incident.py`) racontent un « cover » qui a ouvert une position longue au lieu d'en fermer une, puis une boucle de rachat. Ce simulateur reproduit ce que le bot *croit* ; un compte de marge réel est plus dangereux.
5. **Stop du short vérifié seulement à la clôture de la bougie** (`high ≥ stop`), puis ordre au marché : aucun stop n'est posé sur l'exchange, et la sortie se fait au prix suivant, pas au prix du stop.
6. **Levier croissant avec le capital** (×2 sous 100, ×3 jusqu'à 500, ×5 au-delà) : le risque augmente après les gains. Sur une position entièrement à ×5, une baisse d'environ 12 % de l'actif suffit à faire tomber le niveau de marge à 40 % (seuil de liquidation supposé ici, non vérifié).
7. **Code mort** : un fragment de `build_grid` copié après un `return` (lignes 237-248).

## Limites assumées

- **Coûts non mesurés.** Frais 0,26 % (valeur du bot, proche des 0,271 % relevés sur une vente réelle), glissement 0,03 %, rollover 0,02 % par 4 h : ce dernier est **une hypothèse** (Kraken indique un taux variable fixé à l'exécution, sans intervalle sur la page consultée). Le seuil de liquidation à 40 % du niveau de marge est lui aussi une hypothèse ; seuls les 2 % de frais de liquidation viennent de la page Kraken. Le tableau de sensibilité montre que la conclusion tient sur toute la plage plausible des frais, mais pas la valeur exacte des pertes.
- **Autre marché que le bot réel.** Données Binance ETH/USDT, alors que le bot trade ETH/EUR sur Kraken.
- **Les paramètres du bot ne sont pas vierges.** Ils ont été choisis par leur auteur avant ce projet, peut-être en voyant ces mêmes années. La fenêtre VAL n'est donc pas une preuve hors échantillon au sens strict : elle écarte l'optimisme de DEV, pas tout biais de sélection.
- **Les paramètres ne sont pas appris.** `--set` les change mais rien ici ne cherche de bonnes valeurs : une recherche devrait être comptée (`--tests-counted`) et jugée sur une fenêtre qu'elle n'a pas vue.
- **Un seul actif, une seule période.** Ce banc d'essai ne dit pas que toute stratégie de grille échoue, seulement que celle-ci n'est pas démontrée sur ces données.
- **Indicateurs sur tout l'historique.** Le bot réel les calcule sur 100 bougies : écart moyen de 0,4 point de RSI mais jusqu'à 5 points au pire sur l'ETH réel (`tests/test_indicators.py`).
- **Exécution à l'ouverture suivante, au prix médian ± glissement.** Pas de carnet d'ordres, pas d'ordres partiels.
- **31 heures manquantes** dans les données réelles (0,05 %), laissées telles quelles et comptées dans le rapport de qualité. Les données fictives n'ont aucun trou.
- **Testé sous Python 3.13 uniquement**, et CI non exécutée : le workflow (Python 3.10 et 3.12) est écrit mais n'a pas tourné.

## Structure

| Fichier | Rôle |
|---|---|
| `src/tradelab/strategy.py` | Stratégie en fonction pure : régime ADX/MA50, grille, short, stops, levier adaptatif |
| `src/tradelab/account.py` | Compte de simulation : marge, rollover, liquidation, positions longue et courte |
| `src/tradelab/costs.py` | Modèle de coûts et mesure des erreurs du calcul d'origine (`legacy_reported_pnl`) |
| `src/tradelab/backtest.py` | Moteur : décision à la clôture, exécution à l'ouverture suivante |
| `src/tradelab/validation.py` | Juge : bootstrap par mois, placebo, Bonferroni, verdict |
| `src/tradelab/carry.py` | Portage du financement : table annuelle, fenêtres, rendement sur capital |
| `src/tradelab/state.py` | Stockage d'état atomique avec sauvegarde |
| `src/tradelab/data.py` | `.npz` → bougies, rééchantillonnage, qualité |
| `src/tradelab/synthetic.py` | Données fictives reproductibles (marche aléatoire, financement inventé) |
| `src/tradelab/options.py` | Ce que l'utilisateur peut choisir : paramètres de la stratégie (validés), niveaux de levier, analyses |
| `src/tradelab/cli.py` | `tradelab backtest`, `tradelab carry` et `tradelab params` |
| `examples/params.example.json` | Exemple de fichier pour `--params` |
| `data/synthetic/` | Les fichiers fictifs lus par défaut |
| `scripts/make_synthetic.py` | Régénère `data/synthetic/` à l'identique |
| `scripts/build_sample.py` | Convertit les archives `.npz` de l'auteur en CSV réels dans `data/real/` (ignoré par git) |

## Tests

```bash
pytest
```

- **Invariant comptable** : sans coût, l'équité est exactement le produit des (1 + exposition × rendement de bougie).
- **Pas de regard vers l'avenir** : modifier les bougies futures ne change pas l'équité passée ; chaque ordre s'exécute à l'ouverture de la bougie suivante.
- **Indicateurs** comparés à des implémentations de Wilder écrites à la main, sur huit séries, et vérifiés causals.
- **Juge** contrôlé par un témoin positif (stratégie clairvoyante : démontrée) et un témoin négatif (bruit : non démontré).
- **Données fictives** : déterministes, sans trou, volatilité annualisée proche de celle demandée, queues épaisses, et fichiers du dépôt identiques à la sortie du générateur.
- **Options** : bornes de chaque paramètre (acceptées) et valeurs juste au-delà (refusées), éléments `--set` mal formés, types, priorité de `--set` sur `--params`, listes de leviers, analyses, décompte de Bonferroni, et la garantie qu'une option invalide arrête le lancement avant toute lecture de données.
- **Mutation** : casser volontairement dix comportements (inventaire non décrémenté, exécution à la clôture, p-value sans correction, ADX mal amorcé, frais nuls, etc.) fait échouer au moins un test à chaque fois. Un test trop faible sur l'amorçage de l'ADX a été renforcé à cette occasion. Le même exercice sur le code des options (43 modifications) a trouvé un contrôle redondant, retiré, et aucun test faible.

## Choix de conception

**Pas de trading réel, volontairement.** Le dépôt ne contient aucun code d'envoi d'ordres, aucune clé, aucun appel réseau. Les mesures ci-dessus montrent que la stratégie n'est pas rentable ; la brancher à de l'argent n'aurait aucun sens. Pour un autre usage, les ordres produits par `decide()` (`open_long`, `close_short`, …) se traduisent directement en appels à un courtier.

**Verdict conservateur.** Trois conditions simultanées, correction de Bonferroni pour le nombre de variantes, au moins 24 mois de validation. Un test qui ne peut pas échouer ne prouve rien : c'est pourquoi le juge est lui-même testé sur des cas dont on connaît la réponse.

## Avertissement

Outil d'analyse, pas un conseil financier. Les résultats décrivent le passé sur des données et des hypothèses de coûts précises ; ils ne garantissent rien pour l'avenir.

## Licence

© 2026 Samuel Mateo. Tous droits réservés.

Le code est publié pour être lu et évalué. Sans accord écrit de l'auteur, il ne peut être ni copié, ni modifié, ni redistribué. Les données réelles évoquées plus haut restent soumises aux conditions de Binance Vision.
