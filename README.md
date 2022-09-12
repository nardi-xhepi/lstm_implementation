# Implémentation LSTM avec NumPy

## Définition formelle d'une cellule LSTM

LSTM (Long-Short Term Memory) est un type particulier de cellule utilisée dans les réseaux de neurones récurrents. La définition formelle utilisée dans cette implémentation est celle donnée dans la page suivante : <a href="https://en.wikipedia.org/wiki/Long_short-term_memory">Définition de LSTM</a>. Brièvement, voici les formules caractérisant la cellule LSTM à un instant t:

<p align="center">
  <img src="https://github.com/nardi-xhepi/lstm_implementation/blob/main/images/def_lstm.jpg"></img>
</p>


## Descente du gradient
On utilise ici l'algorithme de la descente du gradient afin de parvenir à une solution optimale pour le problème que l'on souhaite résoudre. Cet algorithme s'appliquera durant la phase appelée "Backward propagation" de notre réseau de neurones. Tout au long de ce fichier, la constante C exprime la fonction de coût qui permet de calculer l'erreur de l'algorithme lorsqu'il fait une prédiction, le but étant de minimiser cette fonction de coût. On implémentera 5 théorèmes principaux qui sont au coeur de la cellule LSTM afin de lui permettre "d'apprendre" (la démonstration se fait en utilisant la règle de la chaîne):

### Théorème 1
<p align="center">
  <img src="https://github.com/nardi-xhepi/lstm_implementation/blob/main/images/th1.jpg"></img>
</p>

### Théorème 2
<p align="center">
  <img src="https://github.com/nardi-xhepi/lstm_implementation/blob/main/images/th2.jpg"></img>
</p>

### Théorème 3
<p align="center">
  <img src="https://github.com/nardi-xhepi/lstm_implementation/blob/main/images/th3.jpg"></img>
</p>

### Théorème 4
<p align="center">
  <img src="https://github.com/nardi-xhepi/lstm_implementation/blob/main/images/th4.jpg"></img>
</p>

### Théorème 5
<p align="center">
  <img src="https://github.com/nardi-xhepi/lstm_implementation/blob/main/images/th5.jpg"></img>
</p>


