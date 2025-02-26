import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from micrograd.engine.value import Value
from micrograd.nn.mlp import MLP

def train_moons_classifier():
    """
    Entraîne un réseau de neurones sur le problème de classification des deux lunes.
    """
    # Fixe les graines aléatoires pour la reproductibilité
    np.random.seed(1337)
    random.seed(1337)
    
    # Génération des données (des points dans un plan)
    X, y = make_moons(n_samples=100, noise=0.1)
    y = y*2 - 1  # Convertit les labels 0/1 en -1/1
    
    # Visualisation des données
    plt.figure(figsize=(5,5))
    plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap='jet')
    plt.title("Données d'entraînement")
    plt.show()
    
    # Initialisation du réseau de neurones
    model = MLP(2, [16, 16, 1])
    print(f"Nombre de paramètres: {len(model.parameters())}")
    
    # Fonction de perte
    def loss(batch_size=None):
        # Chargement des données en ligne
        if batch_size is None:
            Xb, yb = X, y
        else:
            ri = np.random.permutation(X.shape[0])[:batch_size]
            Xb, yb = X[ri], y[ri]
        
        # Convertit les entrées en objets Value
        inputs = [list(map(Value, xrow)) for xrow in Xb]
        
        # Applique le modèle pour obtenir les scores
        scores = list(map(model, inputs))
        
        # Calcul de la perte avec relu
        losses = [(1 + -yi * scorei).relu() for yi, scorei in zip(yb, scores)]
        data_loss = sum(losses) * (1.0 / len(losses))
        
        # Régularisation L2 (évite le surapprentissage)
        alpha = 1e-4
        reg_loss = alpha * sum((p * p for p in model.parameters()))
        total_loss = data_loss + reg_loss
        
        # Calcul de la précision
        accuracy = [(yi > 0) == (scorei.data > 0) for yi, scorei in zip(yb, scores)]
        return total_loss, sum(accuracy) / len(accuracy)
    
    # Entraînement du modèle
    for k in range(100):
        # Passe avant
        total_loss, acc = loss()
        
        # Passe arrière
        for p in model.parameters():
            p.grad = 0.0
        total_loss.backward()
        
        # Mise à jour (descente de gradient)
        learning_rate = 1.0 - 0.9*k/100
        for p in model.parameters():
            p.data -= learning_rate * p.grad
        
        if k % 10 == 0:
            print(f"Étape {k}: perte {total_loss.data:.6f}, précision {acc*100:.1f}%")
    
    # Visualisation de la frontière de décision
    h = 0.25
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Xmesh = np.c_[xx.ravel(), yy.ravel()]
    inputs = [list(map(Value, xrow)) for xrow in Xmesh]
    scores = list(map(model, inputs))
    Z = np.array([s.data > 0 for s in scores])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Frontière de décision du modèle entraîné")
    plt.show()
    
    return model

if __name__ == "__main__":
    model = train_moons_classifier()
    print("Entraînement terminé avec succès!") 