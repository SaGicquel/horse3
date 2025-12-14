# Calibration Artifacts

Ce dossier contient les artefacts de calibration du modèle de probabilités.

## Structure

```
calibration/
├── README.md                       # Ce fichier
├── health.json                     # Status de santé du système
├── scaler_temperature_*.pkl        # Modèle de température softmax
├── calibrator_platt_*.pkl          # Calibrateur Platt (régression logistique)
├── calibrator_isotonic_*.pkl       # Calibrateur Isotonique
├── blender_*.pkl                   # Modèle de blend logit-space
├── calibration_report_*.json       # Rapports détaillés
├── calibration_reliability.png     # Graphique reliability diagram
└── calibration_ece_bins.png        # Graphique ECE par bin
```

## Fichiers clés

### `health.json`
Contient l'état de santé du système de calibration:
- `last_calibration`: Date de la dernière calibration
- `temperature`: Température optimale du softmax
- `alpha`: Poids du blend modèle/marché
- `metrics`: Brier score, ECE, Log Loss
- `profit_flat/kelly`: Simulations de profit

### Artefacts `.pkl`
Fichiers pickle contenant les modèles entraînés.
Format: `{type}_{timestamp}.pkl`

## Commandes CLI

```bash
# Lancer une calibration (30 derniers jours)
python cli.py calibrate --days 30

# Vérifier la santé du système
python cli.py health
```

## Workflow

1. **Calibration hebdomadaire** (dimanche):
   ```bash
   python cli.py calibrate --days 30
   ```

2. **Health check quotidien**:
   ```bash
   python cli.py health
   ```

3. **Recalibration automatique** si:
   - Calibration > 7 jours
   - Brier > 0.25 ou ECE > 0.10

## Métriques cibles

| Métrique | Bon | Acceptable | À surveiller |
|----------|-----|------------|--------------|
| Brier    | <0.15 | <0.20 | >0.25 |
| ECE      | <0.05 | <0.08 | >0.10 |
| Âge      | ≤3j | ≤7j | >7j |
