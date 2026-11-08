# RUNBOOK — attitude-ai

## Guida Operativa Completa per la Campagna Sperimentale

**Progetto:** attitude-ai | Tesi Magistrale in Ingegneria Informatica  
**Scopo:** Confronto LQR vs PPO per il controllo d'assetto di un microsatellite  
**Ultimo aggiornamento:** 18 ottobre 2026

## Procedura corrente: revisione 2026-10-18

### v3: precisione locale senza cambiare le soglie di successo

Il run v2 `run_20261018_170417_025854` ha completato 600064 step a
dt=0.05 s. Il best a 500k arresta il moto ma si ferma a circa 2.072 gradi,
con zero successi su cinque condizioni nominali. Il finale e' a circa
2.097 gradi. I due test delle 17:13 e 17:16 usano lo stesso best e gli
stessi seed: non sono repliche indipendenti.

L'ipotesi v3 e' che il segnale locale possa essere migliorato: la penalita'
angolare v2 a 2.072 gradi e' solo circa -0.00523 per step. Questa osservazione
non dimostra che la reward sia l'unica causa del bias.
Si aggiunge esclusivamente il termine:

```text
precision_reward = 1 / (1 + (angle_rad / deg2rad(2))^2 + (||omega|| / 0.01)^2)
```

E' continuo, limitato fra 0 e 1, vale circa 0.482 al residuo v2 da fermo
e 1 solo al bersaglio con velocita' nulla. Il termine di velocita' impedisce
di ottenere il bonus pieno passando vicino al bersaglio mentre si oscilla.
Restano invariati costo globale di assetto/velocita', costo delle azioni,
bonus `in_target`, durata, reset, iperparametri e selezione fisica del best.
Il premio totale al bersaglio da fermo passa da 1 a 2 per step; la reward
complessiva conserva il vecchio salto del bonus `in_target`, mentre il
nuovo termine locale non ha discontinuita' alla soglia.

Il successo resta errore < 2 gradi E velocita' < 0.01 rad/s per gli ultimi
10 s. Non si allarga la soglia per far passare il precedente modello.
I commenti nel codice riportano risultati, formula sostituita e ragionamento.
Manifest e snapshot identificano la revisione `2026-10-18-local-precision-v3`
e registrano peso e scale del nuovo termine.

Eseguire un nuovo training da zero, allo stesso budget e con gli stessi
seed della v2, per confrontare la nuova reward:

```bash
cd /home/gcontini/PycharmProjects/attitude-ai
source venv/bin/activate

python train.py --timesteps 600000 --dt 0.05 --episode-seconds 60 \
  --checkpoint-freq 100000 --eval-freq 100000 --eval-episodes 3 \
  --eval-seed 2000 --seed 42

python test_and_plot.py --model best --episodes 5 --seed 1000
```

In TensorBoard `control/reward_precision` mostra il nuovo contributo, non
un successo. La decisione continua a basarsi su `validation/success_rate`,
errore e velocita' finali e test separato. Primo criterio operativo:
almeno 4 successi nominali su 5, senza oscillazione persistente. Non c'e'
garanzia che la v3 raggiunga questo obiettivo con 600k step.
I seed 1000-1004 sono gia' stati usati per lo sviluppo: per una conferma
finale da tesi usare in seguito condizioni/seed nuovi e piu' training
indipendenti, non soltanto quelli ripetutamente esaminati durante il tuning.

### Storico v2: pilota con campionamento a 0.05 s e selezione fisica del modello

Il pilota `run_20261018_164350_920143` (301056 step, dt=0.1 s) ha
mostrato oscillazioni persistenti: coppie X/Y circa +/-0.05 N m che
invertono segno ogni campione, velocita' residua circa 0.355 rad/s e
errore nominale circa 2.52 gradi, con zero successi su cinque seed.
Il checkpoint 100k aveva meno errore del finale, senza pero' mantenimento.
Una prova diagnostica della stessa policy a dt=0.05 s elimina
l'oscillazione, ma lascia un bias di circa 2.23 gradi: non e' ancora successo.

La v2 rende il passo configurabile e lo propaga da `manifest.json`
a tutti gli scenari, alla baseline LQR, ai tempi di guasto e alle metriche.
Non cambiano reward e iperparametri PPO: e' un nuovo esperimento sul
campionamento, non una correzione gia' dimostrata del bias di puntamento.
Gamma, GAE e rollout restano per-step: cambiando dt cambia anche la loro
durata fisica, quindi questo pilota non isola tutti gli effetti del training.

Eseguire questi comandi, non il pilota v1 riportato nello storico:

```bash
cd /home/gcontini/PycharmProjects/attitude-ai
source venv/bin/activate

python train.py --timesteps 600000 --dt 0.05 --episode-seconds 60 \
  --checkpoint-freq 100000 --eval-freq 100000 --eval-episodes 3 \
  --eval-seed 2000 --seed 42

python test_and_plot.py --model best --episodes 5 --seed 1000
```

Sono 1200 step per episodio e circa 30000 secondi simulati complessivi,
lo stesso tempo fisico di 300k step a 0.1 s; il numero di transizioni raddoppia.
Il modello riparte da zero. PPO puo' completare il rollout oltre il budget
richiesto: leggere `training_summary.json` per conteggio e tempo effettivi.
Non e' stato avviato automaticamente alcun training lungo.

La validazione ogni 100k step usa tre seed separati (2000-2002) a 15 e
120 gradi, senza rumore o guasti, per 60 e 80 s rispettivamente. E' eseguita
anche al termine, dopo l'ultimo aggiornamento PPO. Non aggiorna le statistiche
del normalizzatore di training. Il test finale usa altri seed (1000-1004),
non quelli usati per selezionare il modello.

La selezione e' lessicografica: prima tasso di successo negli ultimi 10 s,
poi minore errore fisico normalizzato medio nella stessa finestra
`angle_deg / 2 + ||omega|| / 0.01`, infine minore integrale della coppia
applicata al quadrato. Un modello chiamato `best` puo' comunque avere
zero successi: significa solo migliore tra quelli valutati.

Nuovi file nella cartella del training:

| File | Contenuto |
|---|---|
| `ppo_satellite_best.zip` | Policy migliore sui seed di validazione |
| `best_vec_normalize.pkl` | Normalizzatore dello stesso istante della policy best |
| `best_validation.json` | Step selezionato, score e metriche per episodio |
| `validation_history.jsonl` | Storico di tutte le valutazioni, inclusa quella finale |

`--model best` carica esclusivamente questa coppia; nessun fallback silenzioso.
`--model final` (default, per compatibilita' CLI) carica invece la coppia finale.
Per confrontarle, dopo il test best eseguire:

```bash
python test_and_plot.py --model final --episodes 5 --seed 1000
```

In TensorBoard leggere `validation/success_rate`, `validation/tail_angle_deg`
e `validation/tail_omega_rad_s`. Il successo resta errore < 2 gradi e
velocita' < 0.01 rad/s per gli ultimi 10 s. Come criterio preliminare
del pilota, richiedere almeno 4 successi nominali su 5 nel test separato;
non basta che la reward migliori o che il modello sia chiamato best.
`--eval-freq 0` disattiva la validazione e non produce un modello best;
e' obbligatorio per il training 10D `--obs-augmented`, non supportato qui.

Le revisioni precedenti richiedono `--allow-legacy-model` esplicito.
Gli output v1 restano conservati e non vanno mescolati statisticamente
con i nuovi. Manifest e snapshot identificano la revisione v2.

### Storico v1: discreto e mantenimento (precedente al secondo pilota)

La campagna storica `test_20261018_131553` / `test_20261018_161240` ha
mostrato RMS PPO di circa 101-105 gradi, senza stabilizzazione, e residuo
LQR nominale di 2.644 gradi. I risultati sono conservati, non sostituiti.
Non dimostrano superiorita' del PPO ne' robustezza ai guasti.

La nuova revisione usa LQR discreto con ZOH a 0.1 s, quaternioni con segno
coerente, reward quadratica anche sulla velocita', episodi a orizzonte fisso,
generatori casuali locali e rumore di coppia mantenuto costante nello step.
Il successo richiede errore < 2 gradi e velocita' < 0.01 rad/s negli ultimi
10 s. Il settling time riguarda la sola soglia angolare, mantenuta per
almeno gli ultimi 10 s: `N/C` significa non convergenza, non fine simulazione.

### Comandi in ordine

```bash
cd /home/gcontini/PycharmProjects/attitude-ai
source venv/bin/activate

# Pilota: nessuna garanzia di convergenza a questo budget.
python train.py --timesteps 300000 --checkpoint-freq 50000 --seed 42
python test_and_plot.py --episodes 5 --seed 1000
```

In un secondo terminale, con lo stesso ambiente attivo:

```bash
tensorboard --logdir=logs
```

Osservare `control/angle_error_deg`, `control/omega_magnitude`,
`control/in_target` e `control/episode_success` oltre a `rollout/ep_rew_mean`.
Lunghezza episodio ora fissa: non deve scendere quando il controllo migliora.
Se non emerge stabilizzazione, diagnosticare prima di aumentare i timestep.

Solo dopo un pilota soddisfacente:

```bash
python train.py --timesteps 1000000 --checkpoint-freq 100000 --seed 42
python test_and_plot.py --episodes 20 --seed 1000

python train.py --timesteps 2000000 --checkpoint-freq 100000 --seed 42 \
  --domain-rand --sensor-noise 0.005
python test_and_plot.py --episodes 20 --seed 1000
```

Ogni training parte da zero; non e' una ripresa del precedente. I timestep
effettivi possono essere arrotondati da PPO al rollout successivo e sono
registrati in `training_summary.json`. Ripetere le due configurazioni con
altri seed di training (ad esempio 43-46) per stimare la variabilita' fra
policy; piu' episodi dello stesso modello non sostituiscono training indipendenti.

Per selezionare un training specifico senza modificare `latest_run.txt`:

```bash
python test_and_plot.py --run models/run_DATA_ORA_MICROSECONDI \
  --episodes 20 --seed 1000
```

Non usare `--obs-augmented` con questa campagna: il modello 10D viene rifiutato
esplicitamente. I modelli precedenti alla revisione sono rifiutati per default;
per sola analisi esplorativa:

```bash
python test_and_plot.py --run models/run_20261018_151238 \
  --allow-legacy-model --episodes 5 --seed 1000
```

Questo NON riproduce la vecchia baseline: LQR e metriche sono aggiornati.
Non mescolare risultati di revisioni diverse nella stessa media.
I guasti sono applicati solo in valutazione: si misura robustezza a guasti
non visti, non adattamento online o apprendimento esplicito dei guasti.

### Output e storico riproducibile

Ogni nuovo run ha timestamp con microsecondi e una cartella distinta.
`manifest.json` registra revisione, parametri, seed, versioni delle librerie,
commit di partenza e hash SHA-256 dei sorgenti; `source/` contiene una copia
esatta dei moduli Python, delle dipendenze dichiarate e il diff rispetto a HEAD.
Non vengono creati commit automaticamente.

I test registrano anche percorso/hash del modello e normalizzatore e il
manifest del training. Nessun fallback se manca `vec_normalize.pkl`.

| Output in `tests/test_.../` | Significato |
|---|---|
| `metriche_comparative.csv` / `.txt` | Metriche delle traiettorie illustrate, seed esplicito |
| `robustezza_sweep.csv` | Dati numerici dei due sweep illustrativi, singolo seed |
| `episodi.csv` | Metriche di ogni replica e seed, nei tre scenari |
| `statistiche.csv` | Media, deviazione standard campionaria, numero valori validi/totali |
| `*_seed*.npz` | Serie temporali complete delle repliche |
| `manifest.json`, `source/` | Identita' del modello, configurazione e sorgenti |

Le serie `.npz` contengono N+1 stati/tempi, incluso l'istante finale,
e N coppie applicate nei rispettivi intervalli ZOH.

`success` medio in `statistiche.csv` e' il tasso di successo, da 0 a 1.
La media del settling time esclude i fallimenti: leggere sempre
`n_valid/n_total` insieme al tasso di successo. Con una sola replica la
deviazione standard e' `N/A`. Gli sweep restano illustrativi, non bande
statistiche multi-seed. La metrica storicamente chiamata "Energia" e'
l'integrale della coppia applicata al quadrato, in N^2 m^2 s, non joule.
Il rumore sensori e' una deviazione standard additiva sulle componenti
quaternioniche e sulle velocita' in rad/s: non una percentuale fisica uniforme.

Regressioni locali senza installare nuovi strumenti:

```bash
python -m unittest test_control_regressions -v
```

## Archivio operativo precedente (agosto 2026)

Il resto del documento e' conservato per lo storico. In caso di differenze,
valgono procedura, metriche e avvertenze della revisione qui sopra.

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
