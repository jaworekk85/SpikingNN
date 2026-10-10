# Working in SpikingNN

Read `SNN_AGENT_RESEARCH_SPEC.md` before substantial changes. It is the authoritative
development specification. Check `STATUS.md` for the active phase and pending review.

- Work only within the requested phase. Stop at phase boundaries and update `STATUS.md`.
- Phase 0 contains architecture documentation and a package skeleton only.
- Keep learning rules separate from models, datasets and simulation backends.
- Do not implement research directions or the Jülich pilot without their requested phase.
- Use tiny CPU configurations and synthetic data for tests; never require dataset downloads.
- Keep simulation energy proxies separate from measured hardware energy.
- Preserve the original specification; record decisions and ambiguities in `STATUS.md`.
- Do not commit generated results, cached datasets, checkpoints or credentials.
