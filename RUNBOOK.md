# RUNBOOK — attitude-ai

## Guida Operativa Completa per la Campagna Sperimentale

**Progetto:** attitude-ai | Tesi Magistrale in Ingegneria Informatica  
**Scopo:** Confronto LQR vs PPO per il controllo d'assetto di un microsatellite  
**Ultimo aggiornamento:** Agosto 2026

---

## ⚡ Quick Start (Prima Volta)

```bash
# 1. Entra nella cartella del progetto
cd /home/gcontini/PycharmProjects/attitude-ai

# 2. Attiva l'ambiente virtuale
source venv/bin/activate

# 3. Installa le dipendenze (solo la prima volta)
pip install -r requirements.txt

# 4. Avvia il training PPO (~2-4 ore)
python train.py

# 5. Esegui la campagna sperimentale completa
python test_and_plot.py
```

---

## 📋 Flusso Completo Dettagliato

### FASE 0 — Setup Ambiente (solo la prima volta)

```bash
cd /home/gcontini/PycharmProjects/attitude-ai
source venv/bin/activate
pip install -r requirements.txt
```

Verifica installazione:
```bash
python -c "import stable_baselines3, gymnasium, numpy, matplotlib, scipy, torch; print('OK')"
```

---

### FASE 1 — Training dell'Agente PPO

#### Training Base (configurazione standard)

```bash
python train.py
```

Questo produce un agente addestrato con le impostazioni di default:
- 1.000.000 timesteps
- Learning rate: 3×10⁻⁴
- Nessun disturbo, nessuna randomizzazione

#### Training Avanzato (con funzionalità estese)

Per un agente più robusto, attivare le funzionalità avanzate:

```bash
# Training con Domain Randomization (±10% inerzia) e disturbi ambientali
python train.py --domain-rand --disturbances

# Training con rumore sui sensori (σ = 0.01)
python train.py --sensor-noise 0.01

# Training completo con tutte le funzionalità
python train.py --domain-rand --disturbances --sensor-noise 0.005

# Training più lungo per maggiore convergenza
python train.py --timesteps 2000000 --domain-rand --disturbances --sensor-noise 0.005

# Training con observation augmentation (ATTENZIONE: cambia dimensione osservazione!)
python train.py --obs-augmented --timesteps 1500000
```

**Parametri da riga di comando disponibili:**

| Flag | Default | Descrizione |
|------|---------|-------------|
| `--timesteps, -t` | 1.000.000 | Numero totale di timesteps |
| `--checkpoint-freq, -c` | 50.000 | Frequenza salvataggio checkpoint |
| `--learning-rate, -lr` | 3e-4 | Learning rate di Adam |
| `--domain-rand` | OFF | Attiva Domain Randomization |
| `--sensor-noise` | 0.0 | Deviazione standard rumore sensori |
| `--disturbances` | OFF | Attiva disturbi ambientali |
| `--obs-augmented` | OFF | Attiva observation augmentation (10D) |

**⚠️ Nota importante:** `--obs-augmented` cambia la dimensione dello spazio delle osservazioni da 7 a 10. Un modello addestrato con questa flag NON è compatibile con lo script di test standard. Usare solo se si modifica anche il test di conseguenza.

**Output prodotto:**
```
models/
  latest_run.txt               ← puntatore all'ultimo run
  run_YYYYMMDD_HHMMSS/
    ppo_satellite_final.zip    ← modello finale
    vec_normalize.pkl          ← statistiche di normalizzazione
    ppo_satellite_50000_steps.zip   ← checkpoint intermedi
    ...
logs/
  run_YYYYMMDD_HHMMSS/
    PPO_1/                     ← log TensorBoard
```

---

### FASE 1b — Monitoraggio del Training (in un terminale separato)

```bash
cd /home/gcontini/PycharmProjects/attitude-ai
source venv/bin/activate
tensorboard --logdir=logs
```

Aprire nel browser: **http://localhost:6006**

**Metriche chiave da monitorare:**

| Metrica TensorBoard | Significato | Andamento atteso |
|---|---|---|
| `rollout/ep_rew_mean` | Reward media per episodio | Deve salire progressivamente |
| `rollout/ep_len_mean` | Durata media episodi | Diminuisce se l'agente converge prima |
| `train/value_loss` | Loss della value function | Scende e si stabilizza |
| `train/entropy_loss` | Entropia della policy | Cala gradualmente |
| `train/policy_loss` | Loss della policy | Oscilla ma non esplode |

> 💡 **Per la tesi (Cap. 4):** Screenshot delle curve a ~200k, ~500k e 1M steps.

---

### FASE 2 — Campagna Sperimentale (Capitolo 5 della tesi)

```bash
python test_and_plot.py
```

**Cosa esegue (in ordine):**

1. **Test 1 — Scenario Nominale (§5.1):**
   - Condizione iniziale: 15° di errore
   - Nessun guasto, nessun disturbo
   - Confronto diretto LQR vs PPO in regime lineare
   - Output: `test1_nominale.png`

2. **Test 2 — Regime Non Lineare (§5.2):**
   - Condizione iniziale: 120° di errore
   - Dimostra i limiti della linearizzazione dell'LQR
   - Evidenzia la zona dove LQR perde validità (>15°)
   - Output: `test2_non_lineare.png`

3. **Test 3 — Robustezza (§5.3):**
   - **Sweep guasti (3a):** Severità 0%-90% con rumore σ=0.005
   - **Sweep rumore (3b):** Rumore 0%-10% con guasto 50%
   - **Traiettoria singola (3c):** Guasto 70% + rumore, vista temporale
   - Output: `test3_robustezza_sweep.png`, `test3_traiettoria_guasto.png`

4. **Tabella Comparativa (§5.4):**
   - Settling time, RMS error, errore a regime, energia, coppia di picco
   - Output: `metriche_comparative.csv`, `metriche_comparative.txt`

**Output prodotto:**
```
tests/
  test_YYYYMMDD_HHMMSS/
    test1_nominale.png            ← §5.1
    test2_non_lineare.png         ← §5.2
    test3_robustezza_sweep.png    ← §5.3 (grafici sweep)
    test3_traiettoria_guasto.png  ← §5.3 (traiettoria temporale)
    metriche_comparative.csv      ← §5.4 (dati per tabella LaTeX)
    metriche_comparative.txt      ← §5.4 (tabella pre-formattata)
```

---

### FASE 3 — Workflow Consigliato per Risultati Ottimali

#### Strategia raccomandata per la tesi:

```bash
# 1. Training base (benchmark)
python train.py -t 1000000
# → Salva il run_id per confronto

# 2. Training robusto (candidato migliore)
python train.py -t 2000000 --domain-rand --disturbances --sensor-noise 0.005
# → Questo produce un agente più generalizzabile

# 3. Confronto su TensorBoard
tensorboard --logdir=logs
# → Screenshot per Cap. 4

# 4. Seleziona il modello migliore (se necessario)
echo "run_YYYYMMDD_HHMMSS" > models/latest_run.txt

# 5. Genera i risultati per Cap. 5
python test_and_plot.py
# → Tutti i grafici e le tabelle pronti per l'inserimento
```

#### Per confrontare modelli diversi:

```bash
# Punta al primo modello
echo "run_20260822_205736" > models/latest_run.txt
python test_and_plot.py
# → tests/test_AAAAAA/ con risultati del primo modello

# Punta al secondo modello
echo "run_20260825_143000" > models/latest_run.txt
python test_and_plot.py
# → tests/test_BBBBBB/ con risultati del secondo modello

# Confronto visivo delle due cartelle
```

---

## 📊 Mapping Output → Capitoli della Tesi

| File generato | Sezione tesi | Contenuto |
|---|---|---|
| Screenshot TensorBoard (training curves) | Cap. 4 - §4.5 | Convergenza del training |
| `test1_nominale.png` | Cap. 5 - §5.1 | Confronto nominale piccoli angoli |
| `test2_non_lineare.png` | Cap. 5 - §5.2 | Confronto regime non lineare |
| `test3_robustezza_sweep.png` | Cap. 5 - §5.3 | Failure envelope e degradazione |
| `test3_traiettoria_guasto.png` | Cap. 5 - §5.3 | Dettaglio temporale con guasto |
| `metriche_comparative.csv` | Cap. 5 - §5.4 | Tabella quantitativa (importabile in LaTeX) |

---

## 🏗️ Architettura dei Moduli

```
satellite_dynamics.py    → Modello fisico (quaternioni, Eulero, disturbi)
        ↓
satellite_env.py         → Ambiente Gymnasium (obs/actions/reward)
        ↓
train.py                 → Addestramento PPO (Stable-Baselines3)
        ↓
test_and_plot.py         → Campagna sperimentale + grafici

lqr_controller.py       → Baseline classica (confronto)
```

**Funzionalità avanzate (configurabili):**
- Domain Randomization: `SatelliteEnv(domain_randomization=True)`
- Sensor Noise: `SatelliteEnv(sensor_noise_std=0.01)`
- Disturbi ambientali: `SatelliteDynamics(disturbances={...})`
- Observation Augmentation: `SatelliteEnv(observation_augmented=True)`
- Curriculum Learning: `SatelliteEnv(initial_angle_max=np.deg2rad(30))`

---

## 🐛 Troubleshooting

### `ImportError: tqdm/rich not found`
```bash
pip install tqdm rich
```

### `FileNotFoundError: Nessun modello PPO trovato`
Eseguire prima il training:
```bash
python train.py
```

### PPO non converge / risultati peggiori dell'LQR
- Verificare su TensorBoard che `ep_rew_mean` stia migliorando
- Provare un training più lungo: `python train.py -t 2000000`
- Provare con domain randomization: `python train.py --domain-rand`
- Ridurre il learning rate: `python train.py -lr 1e-4`

### Warning "Box low's precision lowered"
Warning non bloccante di Gymnasium. Ignorabile.

### Warning "tight_layout" sui grafici
Warning estetico non bloccante. I grafici sono generati correttamente.

### Training lento
Il training usa CPU per default (ottimale per MLP policy).
Per ridurre i tempi: `python train.py -t 500000`

---

## 📁 Pulizia

```bash
# Elimina tutto e riparte da zero
rm -rf models/ logs/ tests/

# Elimina solo i test mantenendo i modelli
rm -rf tests/

# Elimina solo i log
rm -rf logs/
```

