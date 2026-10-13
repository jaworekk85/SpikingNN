# SpikingNN

Zaczątek frameworka badawczego do spiking neural networks (SNN), lokalnych reguł
uczenia oraz adaptacji do niestacjonarnych strumieni danych.

**Stan: fazy 0–4 i raporty gotowe; faza 5 (gotowość sprzętowa: eksport modelu i reguły, sprawdzanie ograniczeń sprzętu) czeka na przegląd.** Pakiet zawiera referencyjny
neuron LIF, enkodery, małe sieci MLP/conv, trzy reguły uczenia (surrogate BPTT, STDP,
reguła trójczynnikowa ze śladami uprawnień), seedowane zmiany danych z harmonogramem,
protokoły offline / prequential / sekwencja zadań, metryki aktywności i adaptacji,
liczenie operacji synaptycznych, proxy energii z jawnymi założeniami, opcjonalną
ograniczoną precyzję wag, niezmienny zapis wyników oraz raporty porównawcze z wielu
seedów (tabele, wykres, blokada porównań runów o różnych ustawieniach). Proxy energii
to oszacowanie, nie pomiar.

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

Małe, syntetyczne dane, CPU, kilkanaście sekund. Służy walidacji pipeline'u, nie jest
wynikiem badawczym. Każde ziarno zapisuje nowy katalog w `results/`.

Wszystkie trzy reguły i raport porównawczy (tabele + wykres) jednym poleceniem:

```powershell
.\.venv\Scripts\python.exe experiments\reference\run.py --config configs\experiments\reference_tiny.yaml --rule all --report
```

Raport trafia do nowego katalogu `results\reference_tiny\report_<czas>_<hash>\`
(`report.md`, `figure.png`, `summary.json`). Pojedyncza reguła: `--rule three_factor`
(lub `stdp`, `surrogate_bptt`). Raport z wcześniej zapisanych runów:

```powershell
.\.venv\Scripts\python.exe experiments\reference\report.py results\reference_tiny
```

Runy o różnym `T`, kodowaniu, rozmiarze modelu lub protokole nie zostaną porównane,
chyba że dodasz `--allow-mismatch` (różnice pojawią się wtedy w raporcie).

## Prawdziwe dane: MNIST i N-MNIST

Te same eksperymenty na prawdziwych danych (podzbiór 6 000 / 1 000 próbek, CPU):

```powershell
# MNIST: ~12 MB, kilka minut dla 3 reguł x 3 seedy
.\.venv\Scripts\python.exe experiments\reference\run.py --config configs\experiments\reference_mnist.yaml --rule all --report
# N-MNIST (kamera zdarzeniowa): ~1.2 GB do pobrania raz, ok. 10-15 min dla 3 reguł x 3 seedy
.\.venv\Scripts\python.exe experiments\reference\run.py --config configs\experiments\reference_nmnist.yaml --rule all --report
```

Dane pobierają się automatycznie przy pierwszym uruchomieniu do `data\` (ignorowane
przez Git) i są sprawdzane sumami MD5. Testy nigdy niczego nie pobierają.
`--hardware configs\hardware\spinnaker2_draft.yaml` dodatkowo eksportuje model i regułę
i sprawdza je względem szkicu ograniczeń SpiNNaker2 (do potwierdzenia z partnerem).

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
