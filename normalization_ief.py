"""
Module de normalisation des images IEF (IsoElectric Focusing)
Adapté pour les gels électrophorétiques avant l'apprentissage automatique.

Méthodes implémentées:
- Flat-field correction: correction des gradients d'éclairage
- Reinhard: normalisation statistique dans l'espace Lab
- Histogram Matching: ajustement des distributions d'intensité
- Macenko: méthode colorimétrique basée sur OD et SVD
"""

import numpy as np
import cv2
from skimage import exposure
from skimage.color import rgb2lab, lab2rgb
from scipy import ndimage
from scipy.linalg import svd
from typing import Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')


class IEFNormalizer:
    """Classe principale pour la normalisation des images IEF"""
    
    def __init__(self):
        self.target_image: Optional[np.ndarray] = None
        self.target_stats: Optional[dict] = None
    
    def set_target(self, target_image: np.ndarray):
        """
        Définit une image cible pour la normalisation
        
        Args:
            target_image: Image de référence (numpy array RGB uint8)
        """
        self.target_image = target_image.astype(np.float32)
        self.target_stats = self._compute_stats(self.target_image)
    
    def _compute_stats(self, img: np.ndarray) -> dict:
        """Calcule les statistiques d'une image dans l'espace Lab"""
        img_lab = rgb2lab(img / 255.0)
        return {
            'mean': np.mean(img_lab, axis=(0, 1)),
            'std': np.std(img_lab, axis=(0, 1))
        }
    
    def flat_field_correction(self, image: np.ndarray, 
                             background: Optional[np.ndarray] = None,
                             sigma: float = 50.0) -> np.ndarray:
        """
        Correction flat-field pour éliminer les gradients d'éclairage
        
        Args:
            image: Image à corriger (numpy array RGB uint8)
            background: Image de fond optionnelle. Si None, estimée par filtrage gaussien
            sigma: Paramètre du filtre gaussien pour estimer le fond (si background=None)
        
        Returns:
            Image corrigée (numpy array RGB uint8)
        """
        img_float = image.astype(np.float32)
        
        if background is None:
            # Estimer le fond par filtrage gaussien
            background = np.zeros_like(img_float)
            for c in range(3):
                background[:, :, c] = ndimage.gaussian_filter(
                    img_float[:, :, c], sigma=sigma
                )
        
        # Éviter la division par zéro
        background = np.clip(background, 1.0, None)
        
        # Correction flat-field
        corrected = (img_float / background) * np.mean(background, axis=(0, 1))
        
        # Normaliser et convertir en uint8
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
        return corrected
    
    def reinhard_normalization(self, image: np.ndarray, 
                              target_image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Normalisation de Reinhard: aligne moyenne et écart-type dans l'espace Lab
        
        Args:
            image: Image source à normaliser (numpy array RGB uint8)
            target_image: Image cible optionnelle. Si None, utilise self.target_image
        
        Returns:
            Image normalisée (numpy array RGB uint8)
        """
        if target_image is None:
            if self.target_image is None:
                raise ValueError("Aucune image cible définie. Utilisez set_target() ou fournissez target_image")
            target_image = self.target_image
        else:
            target_image = target_image.astype(np.float32)
        
        source = image.astype(np.float32) / 255.0
        
        # Convertir en Lab
        source_lab = rgb2lab(source)
        target_lab = rgb2lab(target_image / 255.0)
        
        # Calculer les statistiques
        source_mean = np.mean(source_lab, axis=(0, 1))
        source_std = np.std(source_lab, axis=(0, 1))
        target_mean = np.mean(target_lab, axis=(0, 1))
        target_std = np.std(target_lab, axis=(0, 1))
        
        # Éviter la division par zéro
        source_std = np.clip(source_std, 1e-6, None)
        
        # Normalisation: (source - source_mean) * (target_std / source_std) + target_mean
        normalized_lab = (source_lab - source_mean) * (target_std / source_std) + target_mean
        
        # Convertir en RGB
        normalized_rgb = lab2rgb(normalized_lab)
        
        # Convertir en uint8
        normalized = np.clip(normalized_rgb * 255, 0, 255).astype(np.uint8)
        
        return normalized
    
    def histogram_matching(self, image: np.ndarray,
                          target_image: Optional[np.ndarray] = None,
                          multichannel: bool = True) -> np.ndarray:
        """
        Histogram Matching: ajuste la distribution d'intensité de l'image source
        pour correspondre à celle de l'image cible
        
        Args:
            image: Image source à normaliser (numpy array RGB uint8)
            target_image: Image cible optionnelle. Si None, utilise self.target_image
            multichannel: Si True, traite chaque canal séparément
        
        Returns:
            Image normalisée (numpy array RGB uint8)
        """
        if target_image is None:
            if self.target_image is None:
                raise ValueError("Aucune image cible définie. Utilisez set_target() ou fournissez target_image")
            target_image = self.target_image.astype(np.uint8)
        else:
            target_image = target_image.astype(np.uint8)
        
        # Utiliser skimage pour l'histogram matching
        # channel_axis est utilisé dans les versions récentes de scikit-image
        try:
            # Version récente (channel_axis)
            matched = exposure.match_histograms(
                image, 
                target_image, 
                channel_axis=2 if multichannel else None
            )
        except TypeError:
            # Version ancienne (multichannel)
            matched = exposure.match_histograms(
                image, 
                target_image, 
                multichannel=multichannel
            )
        
        return matched.astype(np.uint8)
    
    def macenko_normalization(self, image: np.ndarray,
                             target_image: Optional[np.ndarray] = None,
                             alpha: float = 1.0,
                             beta: float = 0.15) -> np.ndarray:
        """
        Normalisation de Macenko: méthode colorimétrique basée sur OD (Optical Density) et SVD
        
        Note: Cette méthode est conçue pour les images histopathologiques H&E mais est testée
        ici sur les images IEF pour comparaison.
        
        Args:
            image: Image source à normaliser (numpy array RGB uint8)
            target_image: Image cible optionnelle. Si None, utilise self.target_image
            alpha: Percentile pour l'extraction des pixels (défaut: 1.0)
            beta: Percentile pour l'extraction des pixels (défaut: 0.15)
        
        Returns:
            Image normalisée (numpy array RGB uint8)
        """
        if target_image is None:
            if self.target_image is None:
                raise ValueError("Aucune image cible définie. Utilisez set_target() ou fournissez target_image")
            target_image = self.target_image
        else:
            target_image = target_image.astype(np.float32)
        
        source = image.astype(np.float32)
        target = target_image.astype(np.float32)
        
        # Convertir en Optical Density (OD)
        # OD = -log10(I / I0) où I0 = 255 (intensité maximale)
        # Éviter log(0) en ajoutant un petit epsilon
        epsilon = 1e-6
        source_od = -np.log10((source + epsilon) / 255.0)
        target_od = -np.log10((target + epsilon) / 255.0)
        
        # Extraire les vecteurs de coloration pour l'image source
        source_stain_vectors = self._extract_stain_vectors(source_od, alpha, beta)
        
        # Extraire les vecteurs de coloration pour l'image cible
        target_stain_vectors = self._extract_stain_vectors(target_od, alpha, beta)
        
        # Normaliser l'image source vers l'image cible
        normalized_od = self._apply_stain_normalization(
            source_od, source_stain_vectors, target_stain_vectors
        )
        
        # Convertir de OD vers RGB
        normalized_rgb = 255.0 * np.power(10.0, -normalized_od)
        
        # Clipper et convertir en uint8
        normalized = np.clip(normalized_rgb, 0, 255).astype(np.uint8)
        
        return normalized
    
    def _extract_stain_vectors(self, od_image: np.ndarray, 
                              alpha: float, beta: float) -> np.ndarray:
        """
        Extrait les vecteurs de coloration en utilisant SVD
        
        Args:
            od_image: Image en Optical Density
            alpha: Percentile pour l'extraction
            beta: Percentile pour l'extraction
        
        Returns:
            Vecteurs de coloration (2x3)
        """
        # Reshaper l'image en matrice (pixels x canaux)
        h, w, c = od_image.shape
        od_reshaped = od_image.reshape(-1, c)
        
        # Extraire les pixels dans la plage [beta, alpha] percentile
        # pour chaque canal
        od_flat = od_reshaped.flatten()
        od_sorted = np.sort(od_flat)
        lower = od_sorted[int(len(od_sorted) * beta / 100)]
        upper = od_sorted[int(len(od_sorted) * alpha / 100)]
        
        # Filtrer les pixels dans cette plage
        mask = np.all((od_reshaped >= lower) & (od_reshaped <= upper), axis=1)
        od_filtered = od_reshaped[mask]
        
        if len(od_filtered) < 10:
            # Si pas assez de pixels, utiliser tous les pixels
            od_filtered = od_reshaped
        
        # Normaliser les vecteurs OD
        epsilon_od = 1e-6
        od_norm = od_filtered / (np.linalg.norm(od_filtered, axis=1, keepdims=True) + epsilon_od)
        
        # Appliquer SVD
        try:
            U, S, Vt = svd(od_norm.T, full_matrices=False)
            # Les deux premiers vecteurs principaux sont les vecteurs de coloration
            # Vt est de shape (c, min(c, n_pixels)), on prend les 2 premiers
            if Vt.shape[0] >= 2:
                stain_vectors = Vt[:2, :].T  # Shape (3, 2)
            else:
                # Si pas assez de composantes, utiliser des vecteurs par défaut
                stain_vectors = np.array([[0.65, 0.70, 0.29],
                                         [0.07, 0.99, 0.11]]).T
        except:
            # En cas d'erreur, utiliser des vecteurs par défaut
            stain_vectors = np.array([[0.65, 0.70, 0.29],
                                     [0.07, 0.99, 0.11]]).T
        
        # S'assurer que la shape est (3, 2)
        if stain_vectors.shape[0] != 3:
            stain_vectors = stain_vectors.T
        
        return stain_vectors
    
    def _apply_stain_normalization(self, source_od: np.ndarray,
                                  source_vectors: np.ndarray,
                                  target_vectors: np.ndarray) -> np.ndarray:
        """
        Applique la normalisation de coloration
        
        Args:
            source_od: Image source en OD
            source_vectors: Vecteurs de coloration source
            target_vectors: Vecteurs de coloration cible
        
        Returns:
            Image normalisée en OD
        """
        h, w, c = source_od.shape
        source_od_reshaped = source_od.reshape(-1, c)  # Shape: (n_pixels, 3)
        
        # S'assurer que les vecteurs sont de la bonne shape (3, 2)
        if source_vectors.shape[0] != 3:
            source_vectors = source_vectors.T if source_vectors.shape[1] == 3 else source_vectors
        if target_vectors.shape[0] != 3:
            target_vectors = target_vectors.T if target_vectors.shape[1] == 3 else target_vectors
        
        # Calculer les concentrations de coloration pour la source
        # source_vectors: (3, 2), source_od_reshaped: (n_pixels, 3)
        # On veut résoudre: source_od_reshaped = concentrations @ source_vectors.T
        # concentrations: (n_pixels, 2)
        try:
            # Résoudre: source_od_reshaped @ source_vectors = concentrations @ (source_vectors.T @ source_vectors)
            # Méthode: concentrations = source_od_reshaped @ source_vectors @ inv(source_vectors.T @ source_vectors)
            svt_sv = source_vectors.T @ source_vectors  # (2, 2)
            source_concentrations = source_od_reshaped @ source_vectors @ np.linalg.inv(svt_sv + np.eye(2) * 1e-6)  # (n_pixels, 2)
        except:
            # En cas d'erreur, utiliser une méthode simplifiée (pseudo-inverse)
            try:
                source_concentrations = source_od_reshaped @ np.linalg.pinv(source_vectors.T).T  # (n_pixels, 2)
            except:
                # Dernière méthode de secours: utiliser Reinhard comme fallback
                return source_od
        
        # Appliquer les vecteurs de coloration cibles
        # target_vectors: (3, 2), source_concentrations: (n_pixels, 2)
        target_od_reshaped = source_concentrations @ target_vectors.T  # Shape: (n_pixels, 3)
        
        # Reshaper en image
        target_od = target_od_reshaped.reshape(h, w, c)
        
        return target_od
    
    def normalize_pipeline(self, image: np.ndarray,
                          methods: list = ['flat_field', 'reinhard'],
                          target_image: Optional[np.ndarray] = None,
                          **kwargs) -> np.ndarray:
        """
        Pipeline de normalisation combinant plusieurs méthodes
        
        Args:
            image: Image à normaliser (numpy array RGB uint8)
            methods: Liste des méthodes à appliquer dans l'ordre
                    Options: 'flat_field', 'reinhard', 'histogram_matching', 'macenko'
            target_image: Image cible pour les méthodes qui en ont besoin
            **kwargs: Paramètres additionnels pour les méthodes
        
        Returns:
            Image normalisée (numpy array RGB uint8)
        """
        result = image.copy()
        
        # Définir l'image cible si fournie
        if target_image is not None:
            self.set_target(target_image)
        
        for method in methods:
            if method == 'flat_field':
                sigma = kwargs.get('flat_field_sigma', 50.0)
                result = self.flat_field_correction(result, sigma=sigma)
            
            elif method == 'reinhard':
                result = self.reinhard_normalization(result, target_image=target_image)
            
            elif method == 'histogram_matching':
                result = self.histogram_matching(result, target_image=target_image)
            
            elif method == 'macenko':
                result = self.macenko_normalization(result, target_image=target_image)
            
            else:
                warnings.warn(f"Méthode inconnue: {method}. Ignorée.")
        
        return result


def select_reference_image(images: list, method: str = 'median') -> np.ndarray:
    """
    Sélectionne une image de référence à partir d'une liste d'images
    
    Args:
        images: Liste d'images (numpy arrays)
        method: Méthode de sélection ('median', 'mean', 'first', 'random')
    
    Returns:
        Image de référence
    """
    if method == 'first':
        return images[0]
    
    elif method == 'random':
        return np.random.choice(images)
    
    elif method == 'median':
        # Calculer la médiane pixel par pixel
        stack = np.stack(images, axis=0)
        return np.median(stack, axis=0).astype(np.uint8)
    
    elif method == 'mean':
        # Calculer la moyenne pixel par pixel
        stack = np.stack(images, axis=0)
        return np.mean(stack, axis=0).astype(np.uint8)
    
    else:
        raise ValueError(f"Méthode inconnue: {method}")

