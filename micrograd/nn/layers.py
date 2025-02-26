import random
from micrograd.engine.value import Value

class Neuron:
    """
    Implémente un neurone simple avec une fonction d'activation.
    
    Attributs:
        w: Liste des poids pour chaque entrée
        b: Biais du neurone
        activation: Fonction d'activation à appliquer (relu ou tanh)
    """
    def __init__(self, nin, activation='relu'):
        # Initialisation des poids avec de petites valeurs aléatoires
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = activation
        
    def __call__(self, x):
        # Calcul de la somme pondérée des entrées
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        # Application de la fonction d'activation
        if self.activation == 'relu':
            return act.relu()
        elif self.activation == 'tanh':
            return act.tanh()
        else:
            return act  # Pas d'activation (linéaire)
    
    def parameters(self):
        # Retourne tous les paramètres du neurone (poids et biais)
        return self.w + [self.b]

class Layer:
    """
    Implémente une couche de neurones.
    
    Attributs:
        neurons: Liste des neurones dans la couche
    """
    def __init__(self, nin, nout, activation='relu'):
        # Création des neurones de la couche
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]
    
    def __call__(self, x):
        # Calcul des sorties de tous les neurones pour l'entrée donnée
        return [n(x) for n in self.neurons]
    
    def parameters(self):
        # Retourne tous les paramètres de la couche
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params 