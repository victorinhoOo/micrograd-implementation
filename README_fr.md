# Implémentation de Micrograd avec un test de classification

Ce projet est une implémentation de **Micrograd**, un framework minimaliste pour le calcul des gradients automatiques, réalisé à l'aide du tutoriel d'Andrej Karpathy : https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ <br><br>
L'objectif est de comprendre en profondeur les concepts fondamentaux du **forward pass**, de la **rétropropagation** (backward pass), et de leur application dans un réseau de neurones.

---

## 🧠 Micrograd

Micrograd est un framework simple pour le calcul automatique des gradients (autograd) grâce à la construction d'un graphe. Chaque opération est enregistrée dans ce graphe, permettant de calculer les dérivées nécessaires pour l'optimisation des paramètres dans un modèle d'apprentissage automatique.

Les concepts abordés dans ce projet incluent :
- Les passes avant et arrière (forward et backward pass).
- La construction d'un réseau neuronal simple (MLP - Multi-Layer Perceptron).
- L'apprentissage supervisé avec des données artificielles.
- Une application à une tâche de classification binaire.

---

## 📁 Structure 

Le notebook contient plusieurs sections :
1. **Recréation de Micrograd :** Développement de la classe `Value` pour représenter des scalaires avec des gradients.
2. **Construction d'un réseau neuronal :** Création des classes `Neuron`, `Layer` et `MLP` pour modéliser un perceptron multicouche.
3. **Test de classification :** Application à une tâche simple de classification binaire avec des données artificielles.
4. **Visualisation des résultats :** Affichage de la frontière de décision du modèle.


---

## 🖼️ Visualisation

Voici une visualisation de la frontière de décision du modèle après l'entraînement :

![Frontière de décision](https://github.com/user-attachments/assets/6dc0d339-9e88-4728-8786-c9c5ad22a514)

---
