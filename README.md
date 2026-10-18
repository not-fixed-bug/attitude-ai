# attitude-ai

**Controllo d'assetto satellitare con Deep Reinforcement Learning**

Progetto di Tesi Magistrale in Ingegneria Informatica  
Confronto tra controllore classico (LQR) e agente Deep RL (PPO)

---

## 📋 Descrizione del Progetto

Questo progetto implementa un simulatore completo in **pure Python** per il controllo d'assetto di un microsatellite spaziale. L'obiettivo è confrontare quantitativamente due approcci di controllo:

1. **LQR (Linear Quadratic Regulator)**: Controllore classico basato sulla teoria del controllo ottimo
2. **PPO (Proximal Policy Optimization)**: Agente di Deep Reinforcement Learning

Il simulatore include:
- ✅ Dinamica realistica del satellite (equazioni di Eulero + cinematica quaternionica)
- ✅ Ambiente conforme a Gymnasium per il training RL
- ✅ Iniezione di guasti per test di robustezza (Fault Tolerance)
- ✅ Visualizzazione comparativa ad alta risoluzione
- ✅ Commenti accademici dettagliati in italiano

---

## 🗂️ Struttura del Progetto

```
attitude-ai/
│
├── requirements.txt           # Dipendenze del progetto
├── satellite_dynamics.py      # Dinamica fisica del satellite
├── satellite_env.py           # Ambiente Gymnasium per RL
├── lqr_controller.py          # Controllore LQR classico
├── train.py                   # Script di training dell'agente PPO
├── test_and_plot.py           # Test comparativo e generazione grafici
├── RUNBOOK.md                 # Guida operativa: ordine dei comandi da eseguire
│
├── models/                    # (Generata da train.py — un run per ogni training)
│   ├── latest_run.txt         # Puntatore all'ultimo run addestrato
│   ├── run_20260821_153000/   # Cartella dedicata al singolo training
│   │   ├── ppo_satellite_final.zip
│   │   ├── vec_normalize.pkl
│   │   └── ppo_satellite_*_steps.zip   # Checkpoint periodici
│   └── run_20260821_190000/   # Un altro training successivo (non sovrascrive il precedente)
│       └── ...
│
├── logs/                      # (Generata da train.py — log TensorBoard per run)
│   ├── run_20260821_153000/
│   └── run_20260821_190000/
│
└── tests/                     # (Generata da test_and_plot.py — un run per ogni test)
    ├── test_20260821_160000/
    │   ├── comparazione_guasto.png   # Figura comparativa (400 DPI)
    │   └── metriche.txt               # Report testuale delle metriche
    └── test_20260821_200000/
        └── ...
```

> **Nota sull'organizzazione dei run:** ogni esecuzione di `train.py` genera un
> identificativo univoco basato su timestamp (`run_AAAAMMGG_HHMMSS`) e crea una
> sottocartella dedicata sia in `models/` sia in `logs/`, così i training
> successivi non sovrascrivono mai quelli precedenti. Allo stesso modo, ogni
> esecuzione di `test_and_plot.py` crea una sottocartella `tests/test_AAAAMMGG_HHMMSS/`
> con la figura e il report di quello specifico test. Lo script di test individua
> automaticamente l'ultimo modello addestrato tramite `models/latest_run.txt`.

---

## 🚀 Installazione

### 1. Requisiti di Sistema

- Python 3.8 o superiore
- 4 GB RAM minimo (8 GB consigliati)
- ~500 MB spazio disco per checkpoint e log

### 2. Setup Ambiente Virtuale (Consigliato)

```bash
# Creazione ambiente virtuale
python -m venv venv

# Attivazione
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. Installazione Dipendenze

```bash
pip install -r requirements.txt
```

---

## 📖 Guida all'Uso

> Per la guida operativa completa con tutti i comandi nell'ordine corretto, vedere **[RUNBOOK.md](RUNBOOK.md)**.

### Step 1: Training dell'Agente PPO

Addestra l'agente di Deep RL per 1 milione di timesteps (~2-4 ore):

```bash
python train.py
```

**Output (nella cartella `models/run_AAAAMMGG_HHMMSS/`):**
- `ppo_satellite_final.zip` — modello finale
- `vec_normalize.pkl` — statistiche di normalizzazione
- `ppo_satellite_*_steps.zip` — checkpoint ogni 50.000 timesteps
- `models/latest_run.txt` — puntatore all'ultimo run (usato da `test_and_plot.py`)

**Monitoraggio in tempo reale (in un secondo terminale):**
```bash
tensorboard --logdir=logs
# Apri http://localhost:6006 nel browser
# Tutti i run appaiono sovrapposti per confronto diretto
```

### Step 2: Test Comparativo e Visualizzazione

Carica automaticamente l'ultimo modello addestrato ed esegue la simulazione comparativa LQR vs PPO con iniezione di guasto:

```bash
python test_and_plot.py
```

**Output (nella cartella `tests/test_AAAAMMGG_HHMMSS/`):**
- `comparazione_guasto.png` — figura comparativa ad alta risoluzione (400 DPI)
- `metriche.txt` — report testuale con tabella delle metriche quantitative
- Metriche stampate anche su console

### Parametri del Test

| Parametro | Valore | Descrizione |
|-----------|--------|-------------|
| Durata simulazione | 60 s | Tempo totale della manovra |
| Passo temporale | 0.1 s | Risoluzione dell'integrazione |
| Errore iniziale | ~120° | Assetto completamente errato |
| Guasto iniettato | t = 20 s | Momento del malfunzionamento |
| Tipo di guasto | 70% riduzione asse X | Perdita efficacia attuatore |

---

## 🔬 Dettagli Tecnici

### Dinamica del Satellite

Il simulatore implementa le equazioni complete del moto rotazionale:

**Cinematica dei Quaternioni:**
```
dq/dt = 0.5 * Ω(ω) * q
```

**Equazioni di Eulero:**
```
J * dω/dt = -ω × (J*ω) + τ
```

dove:
- `q`: quaternione di orientamento [q₀, q₁, q₂, q₃]
- `ω`: velocità angolare [ωₓ, ωᵧ, ωᵤ] [rad/s]
- `J`: matrice di inerzia [kg·m²]
- `τ`: coppie di controllo [N·m]

### Controllore LQR

Minimizza il funzionale di costo quadratico:

```
J = ∫₀^∞ (x^T Q x + u^T R u) dt
```

Legge di controllo ottimale: `u = -K*x`

Matrice di guadagno calcolata risolvendo l'equazione di Riccati:
```
A^T P + P A - P B R⁻¹ B^T P + Q = 0
```

### Agente PPO

**Architettura:**
- Policy network: MLP (Multi-Layer Perceptron)
- Value network: condivisa con la policy
- Spazio azioni: Box continuo [-1, 1]³
- Spazio osservazioni: [quaternione, velocità angolari] ∈ ℝ⁷

**Funzione di Reward:**
```python
reward = -10.0 * (θ_error)²  # Errore di assetto
         -0.01 * ||τ||²      # Consumo energetico
         +100.0              # Bonus al raggiungimento obiettivo
```

---

## 📊 Metriche di Valutazione

Il test comparativo calcola automaticamente:

1. **Errore RMS** [gradi]: Accuratezza media del controllo
2. **Tempo di Settling (5%)** [s]: Rapidità di convergenza
3. **Overshoot Massimo** [gradi]: Stabilità della manovra
4. **Energia Totale** [N²·m²·s]: Efficienza energetica

---

## 🎓 Riferimenti Teorici

### Dinamica Spaziale
- Wertz, J.R. (1978). *Spacecraft Attitude Determination and Control*
- Wie, B. (1998). *Space Vehicle Dynamics and Control*

### Controllo Ottimo
- Kirk, D.E. (2004). *Optimal Control Theory: An Introduction*
- Anderson, B.D. & Moore, J.B. (1990). *Optimal Control: Linear Quadratic Methods*

### Deep Reinforcement Learning
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*
- Lillicrap, T.P. et al. (2015). *Continuous Control with Deep RL*

---

## 🛠️ Troubleshooting

### Problema: Training molto lento

**Soluzione:** 
- Verificare l'installazione PyTorch con supporto GPU
- Ridurre `n_steps` in `train.py` (trade-off: convergenza più lenta)

### Problema: `test_and_plot.py` non trova il modello

**Soluzione:**
```bash
# Verificare esistenza del modello
ls models/ppo_satellite_final.zip

# Se mancante, eseguire nuovamente il training
python train.py
```

### Problema: Errori di normalizzazione durante il test

**Soluzione:**
- Assicurarsi che `vec_normalize.pkl` sia presente in `models/`
- Non modificare i parametri dell'ambiente tra training e test

---

## 📝 Note per la Tesi

### Risultati Attesi

**Prima del guasto (0-20s):**
- Entrambi i controllori dovrebbero convergere rapidamente
- LQR potrebbe essere leggermente più veloce per piccoli errori
- PPO potrebbe mostrare comportamento più smooth

**Dopo il guasto (20-60s):**
- LQR: Possibile degrado prestazioni (modello lineare limitato)
- PPO: Potenziale maggiore robustezza (appreso da variabilità)

### Possibili Estensioni

1. **Multi-guasto**: Test con guasti su più assi
2. **Disturbi esterni**: Aggiunta di coppie stocastiche (gravità gradiente, pressione solare)
3. **Hardware-in-the-Loop**: Integrazione con simulatori tipo Gazebo
4. **Transfer Learning**: Test su satelliti con inerzie diverse

---

## 📄 Licenza

Questo progetto è sviluppato per scopi accademici nell'ambito di una Tesi Magistrale.

---

## 👤 Autore

Progetto **attitude-ai**  
Tesi Magistrale in Ingegneria Informatica  
Anno Accademico 2025/2026

---

## 🙏 Ringraziamenti

- **Gymnasium**: Framework RL standard
- **Stable-Baselines3**: Implementazioni PPO state-of-the-art
- **SciPy**: Integratori numerici e algebra lineare
- **Matplotlib**: Visualizzazioni scientifiche

---

**Per domande o problemi, consultare i commenti dettagliati nel codice sorgente.**

