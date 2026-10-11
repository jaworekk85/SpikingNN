# SpikingNN

Zaczątek frameworka badawczego do spiking neural networks (SNN), lokalnych reguł
uczenia oraz adaptacji do niestacjonarnych strumieni danych.

**Stan: faza 0 — propozycja architektury i szkielet pakietu.** Obecny pakiet można
zainstalować i zaimportować; nie zawiera jeszcze neuronów, treningu ani działających
eksperymentów. Kolejne fazy wymagają przeglądu zgodnie ze specyfikacją.

- [Specyfikacja projektu](SNN_AGENT_RESEARCH_SPEC.md)
- [Architektura, kontrakty i zależności](docs/research_framework.md)
- [Status i otwarte decyzje](STATUS.md)

## Instalacja szkieletu

Python 3.11 lub nowszy. Polecenia dla PowerShell, z katalogu repozytorium:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -c "import snn_research; print(snn_research.__version__)"
```

Pakiet w fazie 0 nie potrzebuje PyTorch ani danych. Zależności do symulacji zostaną
dodane wraz z ich pierwszym wykorzystaniem. Plan przewiduje PyTorch, YAML i opcjonalny
adapter snnTorch.

## Struktura

```text
src/snn_research/  moduły frameworka (obecnie puste pakiety)
configs/          miejsce na konfiguracje YAML
experiments/      miejsce na eksperyment walidacyjny reference
docs/             propozycja architektury i założenia
tests/            plan testów do kolejnych faz
results/          lokalne, ignorowane przez Git wyniki
```

Każda faza kończy się aktualizacją `STATUS.md` i przeglądem. Faza 1 obejmie neuron
LIF, małe modele, kodowanie, surrogate-gradient training, metryki aktywności oraz
zapis wyników. Badania i pilot Jülich pozostają późniejszymi etapami.

Liczba spike'ów, operacje synaptyczne, czas wykonania i zmierzona energia to osobne
wielkości. Proxy energii musi zawierać jawne założenia; pomiar energii wymaga metadanych
z backendu sprzętowego.
