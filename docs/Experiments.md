# Experiments Log – Living-Agent Evolution Sim

This file is the main log of all simulation **experiments**  
("experiências" – Portuguese, Portugal · "Versuche" – Swiss German, Zürich).

Each run gets an ID so it’s easy to refer back to a specific world state, JSON snapshot, or code version.

---

## How to use this log

For every serious run, add a section like:

```text
### Experiment ID: EXP-YYYYMMDD-XX
Date: 2025-12-03
Code version: (git commit hash or tag)
World file: (saved JSON filename if applicable)

Goal:
- What are you trying to see or test?

Setup:
- Grid size:
- Initial agents:
- Food spawn behavior:
- Any code tweaks (mutation rate, reproduction threshold, etc.):

Procedure:
- How long did you run it? (ticks / real-time)
- Did you intervene? (reset, change speed, etc.)

Observations:
- What did you notice? (trait explosions, crashes, interesting patterns)

Outcome:
- Did you reach the goal? What surprised you?

Next steps:
- What to change for the next experiment?
