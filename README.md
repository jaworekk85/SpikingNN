# SpikingNN

Framework badawczy do spiking neural networks (SNN), lokalnych reguł uczenia oraz
adaptacji do niestacjonarnych strumieni danych. Docelowo: pilot dla Forschungszentrum
Jülich (SpiNNaker2) — najpierw wyniki w symulacji, potem sprzęt u partnera.

**Stan:** fazy 0–5 gotowe (aktualny stan i następne kroki: [STATUS.md](STATUS.md)).
Wyniki to walidacja pipeline'u, nie wyniki badawcze; proxy energii to oszacowanie,
nie pomiar.

- [Specyfikacja](SNN_AGENT_RESEARCH_SPEC.md) · [Architektura](docs/research_framework.md) ·
  [Reguły uczenia](docs/learning_rules.md) · [Metryki](docs/metrics.md) ·
  [Sprzęt](docs/neuromorphic_hardware.md)

## Instalacja i testy

Python 3.11+. PowerShell, z katalogu repozytorium:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

Testy działają na CPU w kilkanaście sekund, na małych danych syntetycznych i nigdy
niczego nie pobierają. CI (GitHub Actions) uruchamia je na Pythonie 3.11 i 3.12.

## Eksperyment referencyjny

Jeden przebieg (na seed): trening offline -> strumień z zaplanowaną zmianą danych
(przed / w trakcie / po, adaptacja online) -> dokładność na czystym i zmienionym
teście -> operacje synaptyczne i proxy energii -> zapis do nowego katalogu `results\`.

```powershell
# wszystkie reguły + raport porównawczy (tabele, wykres); dane syntetyczne, sekundy
.\.venv\Scripts\python.exe experiments\reference\run.py --config configs\experiments\reference_tiny.yaml --rule all --report
# MNIST (~12 MB) i N-MNIST (kamera zdarzeniowa, ~1.2 GB) — pobierane raz do data\, sprawdzane MD5
.\.venv\Scripts\python.exe experiments\reference\run.py --config configs\experiments\reference_mnist.yaml --rule all --report
.\.venv\Scripts\python.exe experiments\reference\run.py --config configs\experiments\reference_nmnist.yaml --rule all --report
```

- `--rule surrogate_bptt | three_factor | stdp | all` wybiera regułę.
- `--report` zapisuje `results\<eksperyment>\report_<czas>_<hash>\` (`report.md`,
  `figure.png`, `summary.json`); `report.py <katalog>` robi to dla zapisanych runów.
  Runy o różnym `T`, kodowaniu, rozmiarze modelu lub protokole nie zostaną porównane
  bez `--allow-mismatch`.
- `--hardware configs\hardware\spinnaker2_draft.yaml` eksportuje model i regułę
  (`export.json`, `weights.npz`) i sprawdza je względem szkicu ograniczeń SpiNNaker2
  (do potwierdzenia z partnerem).

Konfiguracje (`configs\experiments\*.yaml`) zawierają tylko wartości inne niż
domyślne; każdy run zapisuje pełną konfigurację w swoim `record.json`.

## Strojenie parametrów (bez zaglądania do testu)

```powershell
.\.venv\Scripts\python.exe experiments\reference\tune.py --config configs\experiments\reference_mnist.yaml
```

Strojenie używa zbioru `tune` wydzielonego z danych treningowych i osobnych seedów;
narzędzie odmawia przyjęcia zbioru testowego. Wynik: `results\<eksperyment>\tuning_*\`
(`tuning.json`, `best_overrides.yaml` do przeniesienia do konfiguracji).

## Struktura

```text
src/snn_research/     framework (neurony, modele, reguły, dane, zmiany, protokoły,
                      metryki, energia, zapis, raporty, eksport sprzętowy)
experiments/reference pipeline.py (jeden run), run.py, report.py, tune.py
configs/              experiments/*.yaml, hardware/*.yaml (szkice, do potwierdzenia)
docs/                 architektura, reguły, metryki, sprzęt
tests/                testy CPU (bez pobierania danych)
data/, results/       lokalne, ignorowane przez Git
```

Zasady pracy (dla ludzi i agentów): [AGENTS.md](AGENTS.md). Liczba spike'ów, operacje
synaptyczne, czas wykonania i zmierzona energia to osobne wielkości.
