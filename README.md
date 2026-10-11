# SpikingNN

Zaczątek frameworka badawczego do spiking neural networks (SNN), lokalnych reguł
uczenia oraz adaptacji do niestacjonarnych strumieni danych.

**Stan: faza 1 — fundamenty, czeka na przegląd.** Pakiet zawiera referencyjny neuron
LIF, enkodery, małe sieci MLP/conv, trening surrogate-gradient, metryki aktywności
i niezmienny zapis wyników. Lokalne reguły uczenia i przesunięcia danych to kolejne fazy.

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
