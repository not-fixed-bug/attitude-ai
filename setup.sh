#!/bin/bash
#####################################################################
# Script di installazione automatica per il progetto attitude-ai
# Tesi Magistrale in Ingegneria Informatica
#####################################################################

echo "======================================================================"
echo "INSTALLAZIONE PROGETTO ATTITUDE-AI"
echo "Controllo d'assetto satellitare con Deep Reinforcement Learning"
echo "======================================================================"
echo ""

# Verifica della versione Python
echo "[1/4] Verifica versione Python..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "      Python installato: $python_version"

required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "      ✓ Versione Python compatibile (>= 3.8)"
else
    echo "      ✗ ERRORE: Python 3.8 o superiore richiesto"
    exit 1
fi
echo ""

# Creazione ambiente virtuale
echo "[2/4] Creazione ambiente virtuale..."
if [ -d "venv" ]; then
    echo "      Ambiente virtuale già esistente. Rimozione..."
    rm -rf venv
fi

python3 -m venv venv
echo "      ✓ Ambiente virtuale creato"
echo ""

# Attivazione ambiente virtuale
echo "[3/4] Attivazione ambiente virtuale..."
source venv/bin/activate
echo "      ✓ Ambiente virtuale attivato"
echo ""

# Installazione dipendenze
echo "[4/4] Installazione dipendenze..."
echo "      Questo potrebbe richiedere alcuni minuti..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "      ✓ Tutte le dipendenze installate con successo"
else
    echo "      ✗ ERRORE durante l'installazione delle dipendenze"
    exit 1
fi
echo ""

# Riepilogo
echo "======================================================================"
echo "INSTALLAZIONE COMPLETATA CON SUCCESSO!"
echo "======================================================================"
echo ""
echo "Per iniziare:"
echo "  1. Attiva l'ambiente virtuale:"
echo "     $ source venv/bin/activate"
echo ""
echo "  2. Addestra l'agente PPO:"
echo "     $ python train.py"
echo ""
echo "  3. Esegui i test comparativi:"
echo "     $ python test_and_plot.py"
echo ""
echo "Per ulteriori informazioni, consulta README.md"
echo "======================================================================"

