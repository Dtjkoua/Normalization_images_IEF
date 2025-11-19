# Normalisation Complète du Dataset

## 📋 Description

Ce script normalise **tout le dataset** `Data_paper_TrainVal_Test` avec les **5 méthodes de normalisation**, en préservant la structure originale.

## 🎯 Structure Générée

```
Data_paper_TrainVal_Test_Normalized/
├── flat_field/
│   ├── test/
│   │   ├── AF/
│   │   │   └── [1430 images normalisées]
│   │   ├── AFC/
│   │   │   └── [33 images normalisées]
│   │   ├── AFS/
│   │   ├── AFSC/
│   │   ├── FC/
│   │   ├── FS/
│   │   ├── FSC/
│   │   └── NC/
│   └── train_Val/
│       ├── AF/
│       │   └── [12865 images normalisées]
│       ├── AFC/
│       ├── AFS/
│       ├── AFSC/
│       ├── FC/
│       ├── FS/
│       ├── FSC/
│       └── NC/
│
├── reinhard/
│   ├── test/
│   └── train_Val/
│
├── histogram_matching/
│   ├── test/
│   └── train_Val/
│
├── macenko/
│   ├── test/
│   └── train_Val/
│
├── pipeline_ff_reinhard/
│   ├── test/
│   └── train_Val/
│
└── README.txt
```

## 🚀 Utilisation

### Commande de base

```bash
python normalize_full_dataset_5_methods.py \
    --input Data_paper_TrainVal_Test \
    --output Data_paper_TrainVal_Test_Normalized
```

### Options disponibles

```bash
python normalize_full_dataset_5_methods.py \
    --input Data_paper_TrainVal_Test \
    --output Data_paper_TrainVal_Test_Normalized \
    --methods flat_field reinhard histogram_matching macenko pipeline_ff_reinhard \
    --flat-field-sigma 50.0 \
    --global-target  # Optionnel: utilise une seule image cible globale
```

### Paramètres

- `--input` : Répertoire d'entrée (défaut: `Data_paper_TrainVal_Test`)
- `--output` : Répertoire de sortie (défaut: `Data_paper_TrainVal_Test_Normalized`)
- `--methods` : Liste des méthodes à appliquer (défaut: toutes les 5)
- `--flat-field-sigma` : Paramètre sigma pour flat-field (défaut: 50.0)
- `--global-target` : Utilise une seule image cible globale (au lieu d'une par classe)

## 📊 Méthodes Appliquées

1. **flat_field** : Correction des gradients d'éclairage
2. **reinhard** : Normalisation statistique dans l'espace Lab
3. **histogram_matching** : Ajustement des distributions d'intensité
4. **macenko** : Méthode colorimétrique (non adaptée aux IEF, testée pour comparaison)
5. **pipeline_ff_reinhard** : Combinaison flat-field + Reinhard

## ⚙️ Fonctionnement

### Sélection de l'image cible

Par défaut, le script utilise **une image cible par classe** (première image de chaque classe). Cela garantit une meilleure normalisation car chaque classe a ses propres caractéristiques.

Pour utiliser une seule image cible globale, ajoutez `--global-target`.

### Traitement

Pour chaque méthode :
1. Parcourt tous les splits (`test`, `train_Val`)
2. Pour chaque classe dans chaque split :
   - Sélectionne une image cible
   - Normalise toutes les images de la classe
   - Sauvegarde dans la structure de sortie

### Barre de progression

Le script affiche une barre de progression pour chaque classe, indiquant :
- Le nombre d'images traitées
- La vitesse de traitement (images/seconde)
- Le pourcentage de progression

## 📈 Statistiques du Dataset

### Split `test`
- **AF** : 1430 images
- **AFC** : 33 images
- **AFS** : 143 images
- **AFSC** : 92 images
- **FC** : 2 images
- **FS** : 8 images
- **FSC** : 2 images
- **NC** : 5 images
- **Total** : ~1715 images

### Split `train_Val`
- **AF** : 12865 images
- **AFC** : 297 images
- **AFS** : 1287 images
- **AFSC** : 828 images
- **FC** : 14 images
- **FS** : 70 images
- **FSC** : 16 images
- **NC** : 41 images
- **Total** : ~15418 images

### Total global
- **Total** : ~17133 images
- **Avec 5 méthodes** : ~85665 images normalisées

## ⏱️ Temps d'exécution estimé

- **Par image** : ~0.01-0.02 secondes
- **Pour test** (~1715 images × 5 méthodes) : ~2-3 heures
- **Pour train_Val** (~15418 images × 5 méthodes) : ~20-25 heures
- **Total** : ~22-28 heures

**Note** : Le temps peut varier selon la machine et la taille des images.

## 💾 Espace disque requis

- **Images originales** : ~X GB (à estimer)
- **Images normalisées** : ~X GB × 5 méthodes = ~5X GB

**Recommandation** : Assurez-vous d'avoir suffisamment d'espace disque avant de lancer le script.

## ✅ Vérification

Après l'exécution, vérifiez :

1. **Structure** : Les dossiers sont bien créés
   ```bash
   ls -R Data_paper_TrainVal_Test_Normalized/
   ```

2. **Nombre d'images** : Chaque méthode a le même nombre d'images que l'original
   ```bash
   find Data_paper_TrainVal_Test_Normalized/flat_field -name "*.jpg" | wc -l
   find Data_paper_TrainVal_Test/test -name "*.jpg" | wc -l
   ```

3. **Fichier README** : Vérifiez `Data_paper_TrainVal_Test_Normalized/README.txt`

## 🔧 Dépannage

### Erreur : "Out of memory"
- Réduisez le nombre de méthodes traitées en une fois
- Traitez un split à la fois

### Erreur : "Disk full"
- Vérifiez l'espace disque disponible
- Normalisez une méthode à la fois

### Interruption
- Le script peut être relancé : il ne réécrit pas les images existantes
- Supprimez les dossiers partiels si nécessaire

## 📝 Notes

- Les images sont sauvegardées en **JPEG qualité 95**
- La structure originale est **parfaitement préservée**
- Chaque méthode est dans un **dossier séparé**
- Les noms de fichiers sont **identiques** à l'original

## 🎯 Utilisation pour l'apprentissage

Après normalisation, vous pouvez utiliser chaque méthode séparément :

```python
# Exemple : utiliser le dataset normalisé avec Reinhard
train_dir = "Data_paper_TrainVal_Test_Normalized/reinhard/train_Val"
test_dir = "Data_paper_TrainVal_Test_Normalized/reinhard/test"
```

Ou comparer les performances entre les méthodes lors de l'entraînement.

