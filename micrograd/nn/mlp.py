from micrograd.nn.layers import Layer

class MLP:
    """
    Implémente un perceptron multicouche (réseau de neurones à propagation avant).
    
    Attributs:
        layers: Liste des couches du réseau
    """
    def __init__(self, nin, nouts):
        """
        Initialise un MLP avec les dimensions spécifiées.
        
        Args:
            nin: Nombre d'entrées du réseau
            nouts: Liste des nombres de neurones pour chaque couche
        """
        sz = [nin] + nouts
        self.layers = []
        
        # Création des couches
        for i in range(len(nouts)):
            # Dernière couche avec activation linéaire, autres avec ReLU
            activation = 'linear' if i == len(nouts) - 1 else 'relu'
            self.layers.append(Layer(sz[i], sz[i+1], activation))
    
    def __call__(self, x):
        """
        Propage l'entrée à travers le réseau.
        
        Args:
            x: Entrée du réseau
            
        Returns:
            Sortie du réseau (sortie de la dernière couche)
        """
        for layer in self.layers:
            x = layer(x)
        # Si la sortie est un seul neurone, retourne sa valeur directement
        return x[0] if len(x) == 1 else x
    
    def parameters(self):
        """
        Retourne tous les paramètres du réseau.
        
        Returns:
            Liste de tous les paramètres (poids et biais) du réseau
        """
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params 