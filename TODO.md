- Fare framework plugin per evaluation:

- Scaricare librespeech clean

Indipendemente dal VAE, dobbiamo implementare delle metriche sul test set.
Fare una classe che implementa le metriche, e che prende in input il modello e il test set, e restituisce le metriche.
Le metriche da implementare sono:
- Character error rate (CER)
- Word error rate (WER)
- Mean opinion score (MOS)
- UTMOS (Unsupervised Training Metric for Objective Speech Quality Assessment)