# Run records

Each folder here is the record of one measurement: what was run, on what, and the files that came out, with a `manifest.json` that names the commit, the checkpoint revisions and the hash of each file.
Documents and pages cite these folders by path.

## Naming

A new folder is named `YYYY-MM-DD_what-was-measured`, in lower case with hyphens: the date of the measurement and what it measured (`2026-10-17_drift-monitor`, `2026-10-16_mlx-gemma-31b-4bit`).
The name carries no session code, track code, machine name or person: a reader should be able to tell from it what the folder holds.
Two measurements of one thing on one day add what tells them apart (`_12b`, `_after-restart`), not a letter or a counter.

## Existing folders

Some earlier folders have session codes in their names (`2026-10-14_s11-session`, `2026-10-15_s12-session`).
They keep those names: the evaluation card, the docs and the release notes point at those paths, and a renamed folder would break every pointer.
A new document that cites one of them names the thing it shows (for example "the long-context records of 2026-10-15") and gives the path as the link.

## Contents

Nothing private goes in a run record: no keys, no benchmark item text, no addresses of machines, no personal data (`CONTRIBUTING.md`).
A record that contains a benchmark item's id or label keeps the id and label and not the text.
