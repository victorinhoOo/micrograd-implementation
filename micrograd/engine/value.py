import math

class Value:
    """
    Implémente une valeur scalaire qui garde une trace de son historique de calcul.
    Cette classe est le bloc de base pour construire notre réseau de neurones.
    
    Attributs:
        data: La valeur numérique stockée
        grad: Le gradient de cette valeur (utilisé pour la rétropropagation)
        _backward: Fonction qui calcule le gradient local
        _prev: Ensemble des valeurs parentes dans le graphe de calcul
        _op: L'opération qui a créé cette valeur
        label: Étiquette 
    """
    # Initialise une nouvelle instance de la classe Value.
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    # Retourne une représentation en chaîne de caractères de l'objet Value.
    def __repr__(self):
        return f"Value(data={self.data})"
    
    # Additionne deux objets Value et retourne un nouvel objet Value résultant de l'addition.
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other) # convertit en Value si nécessaire
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    # Additionne deux objets Value et retourne un nouvel objet Value résultant de l'addition.
    def __radd__(self, other): # other + self
        return self + other
    
    # Multiplie deux objets Value et retourne un nouvel objet Value résultant de la multiplication.
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other) # convertit en Value si nécessaire
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    # Multiplie deux objets Value et retourne un nouvel objet Value résultant de la multiplication.
    def __rmul__(self, other):
        return self * other
    
    # Divise deux objets Value et retourne un nouvel objet Value résultant de la division.
    def __truediv__(self, other): 
        return self * other**-1
    
    # Expose un objet Value à une puissance et retourne un nouvel objet Value résultant de l'exponentiation.
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Ne supporte que les exposants entiers ou flottants"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data**(other-1)) * out.grad
        out._backward = _backward
        return out
    
    # Retourne l'opposé d'un objet Value.
    def __neg__(self):
        return -1.0 * self
    
    # Soustrait un objet Value à un autre et retourne un nouvel objet Value résultant de la soustraction.
    def __sub__(self, other):
        return self + (-other)
    
    # Retourne l'opposé d'un objet Value.
    def __rsub__(self, other):
        return other + (-self)
    
    # Applique la fonction ReLU à un objet Value et retourne un nouvel objet Value résultant de l'application de la fonction.
    # La fonction ReLU retourne 0 si la valeur est négative, et la valeur elle-même sinon.
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    # Applique la fonction tangente hyperbolique à un objet Value et retourne un nouvel objet Value résultant de l'application de la fonction.
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    
    # Applique la fonction exponentielle à un objet Value et retourne un nouvel objet Value résultant de l'application de la fonction.
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    # Effectue une rétropropagation à partir d'un objet Value.
    # La rétropropagation est un algorithme utilisé pour calculer les gradients des paramètres du modèle
    # en fonction de la fonction de perte. Elle permet d'ajuster les poids et les biais du réseau de neurones
    # afin de minimiser l'erreur entre les prédictions et les valeurs cibles.
    def backward(self):
        topo = []
        visited = set()

        # Construit un ordre topologique des nœuds 
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        # Pour chaque nœud dans l'ordre topologique, appelle la fonction de rétropropagation
        for node in reversed(topo):
            node._backward()

    """Applique la fonction softmax à la valeur"""
    def softmax(self):
        exp_x = self.exp()
        return exp_x / (1.0 + exp_x) 