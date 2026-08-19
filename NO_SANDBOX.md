# Runtime preference

Non usare mai la sandbox per questo progetto.

Quando un comando deve girare per generare audio, usare accelerazione locale, accedere
alla cache dei modelli, scaricare pesi, o leggere/scrivere risorse fuori dal sandbox,
richiedere esplicitamente l'esecuzione fuori sandbox.
