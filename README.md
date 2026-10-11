# SpikingNN

Zaczątek frameworka badawczego do spiking neural networks (SNN), lokalnych reguł
uczenia oraz adaptacji do niestacjonarnych strumieni danych.

**Stan: faza 2 — lokalne uczenie, czeka na przegląd.** Pakiet zawiera referencyjny
neuron LIF, enkodery, małe sieci MLP/conv, trzy reguły uczenia (surrogate BPTT, STDP,
reguła trójczynnikowa ze śladami uprawnień), metryki aktywności i niezmienny zapis
wyników. Przesunięcia danych i protokoły strumieniowe to kolejna faza.

- [Specyfikacja projektu](SNN_AGENT_RESEARCH_SPEC.md)
- [Architektura, kontrakty i zależności](docs/research_framework.md)
- [Status i otwarte decyzje](STATUS.md)

## Instalacja

Python 3.11 lub nowszy. Polecenia dla PowerShell, z katalogu repozytorium:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

## Eksperyment referencyjny

Małe, syntetyczne dane, CPU, kilka sekund. Służy walidacji pipeline'u, nie jest
wynikiem badawczym. Każde ziarno zapisuje nowy katalog w `results/`.

```powershell
.\.venv\Scripts\python.exe experiments\reference\run.py --config configs\experiments\reference_tiny.yaml
.\.venv\Scripts\python.exe experiments\reference\run.py --config configs\experiments\reference_tiny.yaml --rule three_factor
.\.venv\Scripts\python.exe experiments\reference\run.py --config configs\experiments\reference_tiny.yaml --rule stdp
```

## Struktura

```text
src/snn_research/  moduły frameworka
configs/          konfiguracje YAML
experiments/      eksperyment walidacyjny reference
docs/             architektura i założenia
tests/            testy CPU na danych syntetycznych
results/          lokalne, ignorowane przez Git wyniki
```

Każda faza kończy się aktualizacją `STATUS.md` i przeglądem. Badania i pilot Jülich
pozostają późniejszymi etapami.

Liczba spike'ów, operacje synaptyczne, czas wykonania i zmierzona energia to osobne
wielkości. Proxy energii musi zawierać jawne założenia; pomiar energii wymaga metadanych
z backendu sprzętowego.
