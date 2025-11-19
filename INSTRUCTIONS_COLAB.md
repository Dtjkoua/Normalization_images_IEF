# Instructions pour utiliser le script dans Google Colab

## 📋 Prérequis

1. **Google Colab** : Ouvrez un nouveau notebook sur [colab.research.google.com](https://colab.research.google.com)

2. **Fichiers nécessaires** :
   - `normalization_ief.py` (module principal)
   - `test_5_images_colab.py` (script de test)

## 🚀 Installation et Configuration

### Étape 1 : Installer les dépendances

Dans une cellule Colab, exécutez :

```python
!pip install numpy opencv-python pillow scikit-image scipy matplotlib tqdm
```

### Étape 2 : Télécharger les fichiers

**Option A : Depuis Google Drive**

```python
from google.colab import drive
drive.mount('/content/drive')

# Copier les fichiers depuis Drive
!cp "/content/drive/MyDrive/chemin/vers/normalization_ief.py" /content/
!cp "/content/drive/MyDrive/chemin/vers/test_5_images_colab.py" /content/
```

**Option B : Télécharger depuis GitHub ou autre source**

```python
# Si les fichiers sont sur GitHub
!wget https://raw.githubusercontent.com/votre-repo/normalization_ief.py
!wget https://raw.githubusercontent.com/votre-repo/test_5_images_colab.py
```

**Option C : Coller directement le code**

Vous pouvez aussi copier-coller le contenu de `normalization_ief.py` et `test_5_images_colab.py` dans des cellules Colab.

### Étape 3 : Télécharger votre dataset

**Option A : Depuis Google Drive**

```python
# Si votre dataset est sur Drive
!cp -r "/content/drive/MyDrive/chemin/vers/dataset" /content/
```

**Option B : Depuis un fichier ZIP**

```python
# Télécharger le ZIP
from google.colab import files
uploaded = files.upload()  # Sélectionnez votre fichier ZIP

# Décompresser
!unzip votre_dataset.zip -d /content/
```

**Option C : Depuis une URL**

```python
!wget https://votre-url.com/dataset.zip
!unzip dataset.zip -d /content/
```

### Étape 4 : Configurer le script

Ouvrez `test_5_images_colab.py` et modifiez la section **CONFIGURATION** :

```python
# ============================================================================
# CONFIGURATION - MODIFIEZ ICI
# ============================================================================

# Chemin vers votre dataset dans Colab
DATASET_PATH = "/content/dataset"  # ← MODIFIEZ ICI

# Split à utiliser (TrainSet, ValSet, ou TestSet)
SPLIT = "TestSet"  # ← MODIFIEZ SI NÉCESSAIRE

# Nombre d'images par classe
SAMPLES_PER_CLASS = 5  # ← MODIFIEZ SI NÉCESSAIRE

# Répertoire de sortie
OUTPUT_DIR = "/content/test_normalization_results"  # ← MODIFIEZ SI NÉCESSAIRE

# Paramètre pour flat-field correction
FLAT_FIELD_SIGMA = 50.0  # ← MODIFIEZ SI NÉCESSAIRE
```

### Étape 5 : Vérifier la structure du dataset

Votre dataset doit avoir cette structure :

```
/content/dataset/
├── TrainSet/
│   ├── AF/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── AFC/
│   └── ...
├── ValSet/
│   └── ...
└── TestSet/
    ├── AF/
    ├── AFC/
    └── ...
```

Vérifiez avec :

```python
import os
dataset_path = "/content/dataset"
split = "TestSet"

# Vérifier la structure
split_path = os.path.join(dataset_path, split)
if os.path.exists(split_path):
    classes = [d for d in os.listdir(split_path) 
               if os.path.isdir(os.path.join(split_path, d))]
    print(f"Classes trouvées dans {split}: {classes}")
else:
    print(f"❌ Le répertoire {split_path} n'existe pas")
```

## ▶️ Exécution

### Exécuter le script

Dans une cellule Colab :

```python
# Importer et exécuter
exec(open('test_5_images_colab.py').read())
```

Ou si vous avez collé le code directement :

```python
# Exécuter la fonction main()
main()
```

## 📊 Résultats

Les résultats seront sauvegardés dans `/content/test_normalization_results/` :

```
test_normalization_results/
├── AF/
│   ├── image1_comparison.png
│   ├── image2_comparison.png
│   ├── Flat-field/
│   ├── Reinhard/
│   ├── Histogram_Matching/
│   ├── Macenko/
│   └── Pipeline_(FF+Reinhard)/
├── AFC/
└── ...
```

### Visualiser les résultats dans Colab

```python
from IPython.display import Image, display
import os

# Afficher une image de comparaison
result_dir = "/content/test_normalization_results"
class_name = "AF"  # Modifiez selon votre classe
image_name = "image1_comparison.png"  # Modifiez selon votre image

image_path = os.path.join(result_dir, class_name, image_name)
if os.path.exists(image_path):
    display(Image(image_path))
else:
    print(f"Image non trouvée: {image_path}")
```

### Télécharger les résultats

```python
# Créer un ZIP des résultats
!zip -r /content/results.zip /content/test_normalization_results

# Télécharger
from google.colab import files
files.download('/content/results.zip')
```

## 🔧 Personnalisation

### Changer le nombre d'images par classe

Dans la section CONFIGURATION :

```python
SAMPLES_PER_CLASS = 10  # Au lieu de 5
```

### Tester seulement certaines classes

Modifiez la fonction `main()` pour filtrer :

```python
# Dans main(), avant la boucle for
classes_to_test = ['AF', 'AFC']  # Liste des classes à tester
classes = {k: v for k, v in classes.items() if k in classes_to_test}
```

### Changer le split

```python
SPLIT = "TrainSet"  # Au lieu de "TestSet"
```

## ⚠️ Dépannage

### Erreur : "Module not found"

```python
!pip install --upgrade numpy opencv-python pillow scikit-image scipy matplotlib tqdm
```

### Erreur : "File not found"

Vérifiez les chemins :

```python
import os
print("Dataset existe:", os.path.exists("/content/dataset"))
print("Split existe:", os.path.exists("/content/dataset/TestSet"))
```

### Erreur : "Out of memory"

Réduisez le nombre d'images :

```python
SAMPLES_PER_CLASS = 2  # Au lieu de 5
```

Ou traitez une classe à la fois.

## 📝 Exemple complet dans Colab

```python
# Cellule 1 : Installation
!pip install numpy opencv-python pillow scikit-image scipy matplotlib tqdm

# Cellule 2 : Montage Drive (si nécessaire)
from google.colab import drive
drive.mount('/content/drive')

# Cellule 3 : Copier les fichiers
!cp "/content/drive/MyDrive/normalization_ief.py" /content/
!cp "/content/drive/MyDrive/test_5_images_colab.py" /content/

# Cellule 4 : Copier le dataset
!cp -r "/content/drive/MyDrive/dataset" /content/

# Cellule 5 : Vérifier la structure
import os
print("Classes:", os.listdir("/content/dataset/TestSet"))

# Cellule 6 : Exécuter le script
exec(open('test_5_images_colab.py').read())

# Cellule 7 : Visualiser les résultats
from IPython.display import Image, display
display(Image("/content/test_normalization_results/AF/image1_comparison.png"))
```

## 💡 Astuces

1. **Utiliser GPU** : Dans Colab, allez dans Runtime → Change runtime type → GPU (pour accélérer si vous avez beaucoup d'images)

2. **Sauvegarder sur Drive** : Copiez les résultats sur Drive pour les conserver

```python
!cp -r /content/test_normalization_results /content/drive/MyDrive/
```

3. **Afficher la progression** : Le script affiche déjà la progression, mais vous pouvez ajouter `tqdm` pour les barres de progression

4. **Tester une seule classe** : Modifiez `main()` pour ne traiter qu'une classe

```python
# Dans main(), remplacez la boucle par :
test_class_samples(
    class_name="AF",
    class_dir=classes["AF"],
    output_dir=OUTPUT_DIR,
    max_samples=SAMPLES_PER_CLASS,
    flat_field_sigma=FLAT_FIELD_SIGMA
)
```

