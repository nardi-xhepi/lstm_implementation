# Implémentation LSTM avec NumPy

## Définition formelle d'une cellule LSTM

LSTM (Long-Short Term Memory) est un type particulier de cellule utilisée dans les réseaux de neurones récurrents. La définition formelle utilisée dans cette implémentation est celle donnée dans la page suivante : <a href="https://en.wikipedia.org/wiki/Long_short-term_memory">Définition de LSTM</a>. Brièvement, voici les formules caractérisant la cellule LSTM à un instant t:

<p align="center">
  <img src="https://github.com/nardi-xhepi/lstm_implementation/blob/main/images/def_lstm.jpg"></img>
</p>


## Descente du gradient
On utilise ici l'algorithme de la descente du gradient afin de parvenir à une solution optimale pour le problème que l'on souhaite résoudre. Cet algorithme s'appliquera durant la phase appelée "Backward propagation" de notre réseau de neurones. On implémentera 4 théorèmes principaux qui sont au coeur de la cellule LSTM afin de lui permettre "d'apprendre":

### Théorème 1
<p align="center">
  <img src="https://github.com/nardi-xhepi/lstm_implementation/blob/main/images/def_lstm.jpg"></img>
</p>

### Théorème 2
<p align="center">
  <img src="https://github.com/nardi-xhepi/lstm_implementation/blob/main/images/def_lstm.jpg"></img>
</p>

### Théorème 3
<p align="center">
  <img src="https://github.com/nardi-xhepi/lstm_implementation/blob/main/images/def_lstm.jpg"></img>
</p>

### Théorème 4
<p align="center">
  <img src="https://github.com/nardi-xhepi/lstm_implementation/blob/main/images/def_lstm.jpg"></img>
</p>

