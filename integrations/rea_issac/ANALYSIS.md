# REA-Issac analysis and implementation scope

This scenario is an independent simulation informed by a local REA investigation
of a user-provided Steam installation of The Binding of Isaac: Afterbirth+ on
macOS x86_64. It does not embed, load or redistribute the original game engine,
executables, archives, textures, music, or XML files.

## What REA established

REA/Ghidra inspected the supplied ResourceExtractor's `Archive::open`
(0x100001cd2), `Archive::extract` (0x100001f6c) and `Archive::Unscramble`
(0x100001c3a). The archive header begins with `ARCH000`, followed by a version,
a little-endian table offset and an entry count. Entries hold name hashes,
data offsets, lengths and checksums. A hash-verified copy of the official
extractor recovered an allowlist of configuration XML files in an external
analysis workspace. The installed game files were not modified.

Extractor SHA-256:
`7d0b837bf3826915be1eb3f42cb117a0496ebb1c27aa765b841c61fa587e519b`.
Selected factual values and original configuration hashes are retained in
`integrations/rea_issac/source_facts.json`. REA, Ghidra, the extractor and raw
analysis outputs remain outside this repository and are not runtime dependencies.

The extracted player defaults provide health 6, one bomb and radius 10. Selected
entity defaults inform the chaser, spitter, fly and boss health/radii/contact
damage. The independent Warden uses half the source boss HP and 0.65 times its
radius. Source action constants are recorded for reference; this adapter uses
its own typed action catalog rather than calling an installed-game API.

An earlier static inspection of the game executable yielded implausible
pseudocode. Those outputs were not treated as recovered gameplay logic.
The native game's startup crash was not repaired by this work.

## What this implementation adds

The simulation independently implements an eight-room seeded floor, exploration,
locked combat rooms, collision detection, separate movement and shooting,
projectiles, bombs, health/invulnerability, pickups, upgrades, shops, three
regular enemy behaviors, a boss and terminal outcomes. Attack patterns,
movement, timing, rewards, map generation and item effects are prototype rules,
not a recovered or complete implementation of Isaac. Canvas graphics are drawn
from code; no original sprites are included.

The Honing adapter adds public observations, typed model questions, immutable
task budgets, editable harness instructions, a rule baseline, episode recording,
full-state replay verification, and boss-defeat grading. This establishes a
controllable scenario for decision-model experiments. It does not establish
fidelity to the original game, complete Isaac content, a trained model, or a
measured improvement in model performance.
