NOT FOR MERGING: this file maps every `{{FIG:...}}` slot to the report file and key that fills it; delete it before the records PR is opened.

Where the figures come from: the image run's `report.py` output (`REPORT.md` and `--json report.json`), `latency_table.py`, `fit_T.py`, the Box tree check (`l6.py tree`), and `baseline/metrics_v13.json` for TypeSafe's figures.
In `report.json`, `["<m>/<b>"]` is a model (`b12`, `m16`, `b31`) and the folder suffix of its start (`280`, `1120`); the tuned 12B's single start is shown as `sel`.
A block path like `["S.noul|report"]["noul"]` is `metrics[<part>.<kind>|report][<answer kind>][<metric>]`.
The filler formats a decision metric to 3 decimals, a Box metric to 3 decimals, a latency pair as `p50 / p95` in whole milliseconds, an interval as `diff [lo, hi]` with a sign.

## Slots

| Slot | Source file | Key or path | Note |
| --- | --- | --- | --- |
| `{{FIG:run.power_limit_w}}` | `results/first_contact.json` | `power limit` |  |
| `{{FIG:run.decisio_sha}}` | `results/setup (DEC_SHA in env.sh)` | `` |  |
| `{{FIG:run.m16_budget}}` | `results/m/m16/ (folder name on<B>)` | `` |  |
| `{{FIG:run.date}}` | `results/stage/*.json` | `first stage start date` |  |
| `{{FIG:run.folder}}` | `(name of the run folder)` | `` |  |
| `{{FIG:n.S.noul}}` | `report.json` | `["b12/280"].metrics["S.noul\|report"][<kind>]["n"]` | n is the same for every model (the same items) |
| `{{FIG:m.b12.280.S.noul.accuracy}}` | `report.json` | `report.json["b12/280"].metrics["S.noul\|report"]["noul"]["accuracy"]` |  |
| `{{FIG:m.b12.280.S.noul.logloss}}` | `report.json` | `report.json["b12/280"].metrics["S.noul\|report"]["noul"]["logloss"]` |  |
| `{{FIG:m.b12.280.S.noul.ece}}` | `report.json` | `report.json["b12/280"].metrics["S.noul\|report"]["noul"]["ece"]` |  |
| `{{FIG:m.b12.1120.S.noul.accuracy}}` | `report.json` | `report.json["b12/1120"].metrics["S.noul\|report"]["noul"]["accuracy"]` |  |
| `{{FIG:m.b12.1120.S.noul.logloss}}` | `report.json` | `report.json["b12/1120"].metrics["S.noul\|report"]["noul"]["logloss"]` |  |
| `{{FIG:m.b12.1120.S.noul.ece}}` | `report.json` | `report.json["b12/1120"].metrics["S.noul\|report"]["noul"]["ece"]` |  |
| `{{FIG:m.m16.sel.S.noul.accuracy}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["S.noul\|report"]["noul"]["accuracy"]` |  |
| `{{FIG:m.m16.sel.S.noul.logloss}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["S.noul\|report"]["noul"]["logloss"]` |  |
| `{{FIG:m.m16.sel.S.noul.ece}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["S.noul\|report"]["noul"]["ece"]` |  |
| `{{FIG:m.b31.280.S.noul.accuracy}}` | `report.json` | `report.json["b31/280"].metrics["S.noul\|report"]["noul"]["accuracy"]` |  |
| `{{FIG:m.b31.280.S.noul.logloss}}` | `report.json` | `report.json["b31/280"].metrics["S.noul\|report"]["noul"]["logloss"]` |  |
| `{{FIG:m.b31.280.S.noul.ece}}` | `report.json` | `report.json["b31/280"].metrics["S.noul\|report"]["noul"]["ece"]` |  |
| `{{FIG:m.b31.1120.S.noul.accuracy}}` | `report.json` | `report.json["b31/1120"].metrics["S.noul\|report"]["noul"]["accuracy"]` |  |
| `{{FIG:m.b31.1120.S.noul.logloss}}` | `report.json` | `report.json["b31/1120"].metrics["S.noul\|report"]["noul"]["logloss"]` |  |
| `{{FIG:m.b31.1120.S.noul.ece}}` | `report.json` | `report.json["b31/1120"].metrics["S.noul\|report"]["noul"]["ece"]` |  |
| `{{FIG:n.S.choice}}` | `report.json` | `["b12/280"].metrics["S.choice\|report"][<kind>]["n"]` | n is the same for every model (the same items) |
| `{{FIG:m.b12.280.S.choice.accuracy}}` | `report.json` | `report.json["b12/280"].metrics["S.choice\|report"]["choice"]["accuracy"]` |  |
| `{{FIG:m.b12.280.S.choice.logloss}}` | `report.json` | `report.json["b12/280"].metrics["S.choice\|report"]["choice"]["logloss"]` |  |
| `{{FIG:m.b12.280.S.choice.ece}}` | `report.json` | `report.json["b12/280"].metrics["S.choice\|report"]["choice"]["ece"]` |  |
| `{{FIG:m.b12.1120.S.choice.accuracy}}` | `report.json` | `report.json["b12/1120"].metrics["S.choice\|report"]["choice"]["accuracy"]` |  |
| `{{FIG:m.b12.1120.S.choice.logloss}}` | `report.json` | `report.json["b12/1120"].metrics["S.choice\|report"]["choice"]["logloss"]` |  |
| `{{FIG:m.b12.1120.S.choice.ece}}` | `report.json` | `report.json["b12/1120"].metrics["S.choice\|report"]["choice"]["ece"]` |  |
| `{{FIG:m.m16.sel.S.choice.accuracy}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["S.choice\|report"]["choice"]["accuracy"]` |  |
| `{{FIG:m.m16.sel.S.choice.logloss}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["S.choice\|report"]["choice"]["logloss"]` |  |
| `{{FIG:m.m16.sel.S.choice.ece}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["S.choice\|report"]["choice"]["ece"]` |  |
| `{{FIG:m.b31.280.S.choice.accuracy}}` | `report.json` | `report.json["b31/280"].metrics["S.choice\|report"]["choice"]["accuracy"]` |  |
| `{{FIG:m.b31.280.S.choice.logloss}}` | `report.json` | `report.json["b31/280"].metrics["S.choice\|report"]["choice"]["logloss"]` |  |
| `{{FIG:m.b31.280.S.choice.ece}}` | `report.json` | `report.json["b31/280"].metrics["S.choice\|report"]["choice"]["ece"]` |  |
| `{{FIG:m.b31.1120.S.choice.accuracy}}` | `report.json` | `report.json["b31/1120"].metrics["S.choice\|report"]["choice"]["accuracy"]` |  |
| `{{FIG:m.b31.1120.S.choice.logloss}}` | `report.json` | `report.json["b31/1120"].metrics["S.choice\|report"]["choice"]["logloss"]` |  |
| `{{FIG:m.b31.1120.S.choice.ece}}` | `report.json` | `report.json["b31/1120"].metrics["S.choice\|report"]["choice"]["ece"]` |  |
| `{{FIG:n.S.score}}` | `report.json` | `["b12/280"].metrics["S.score\|report"][<kind>]["n"]` | n is the same for every model (the same items) |
| `{{FIG:m.b12.280.S.score.accuracy}}` | `report.json` | `report.json["b12/280"].metrics["S.score\|report"]["score"]["accuracy"]` |  |
| `{{FIG:m.b12.280.S.score.logloss}}` | `report.json` | `report.json["b12/280"].metrics["S.score\|report"]["score"]["logloss"]` |  |
| `{{FIG:m.b12.280.S.score.ece}}` | `report.json` | `report.json["b12/280"].metrics["S.score\|report"]["score"]["ece"]` |  |
| `{{FIG:m.b12.1120.S.score.accuracy}}` | `report.json` | `report.json["b12/1120"].metrics["S.score\|report"]["score"]["accuracy"]` |  |
| `{{FIG:m.b12.1120.S.score.logloss}}` | `report.json` | `report.json["b12/1120"].metrics["S.score\|report"]["score"]["logloss"]` |  |
| `{{FIG:m.b12.1120.S.score.ece}}` | `report.json` | `report.json["b12/1120"].metrics["S.score\|report"]["score"]["ece"]` |  |
| `{{FIG:m.m16.sel.S.score.accuracy}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["S.score\|report"]["score"]["accuracy"]` |  |
| `{{FIG:m.m16.sel.S.score.logloss}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["S.score\|report"]["score"]["logloss"]` |  |
| `{{FIG:m.m16.sel.S.score.ece}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["S.score\|report"]["score"]["ece"]` |  |
| `{{FIG:m.b31.280.S.score.accuracy}}` | `report.json` | `report.json["b31/280"].metrics["S.score\|report"]["score"]["accuracy"]` |  |
| `{{FIG:m.b31.280.S.score.logloss}}` | `report.json` | `report.json["b31/280"].metrics["S.score\|report"]["score"]["logloss"]` |  |
| `{{FIG:m.b31.280.S.score.ece}}` | `report.json` | `report.json["b31/280"].metrics["S.score\|report"]["score"]["ece"]` |  |
| `{{FIG:m.b31.1120.S.score.accuracy}}` | `report.json` | `report.json["b31/1120"].metrics["S.score\|report"]["score"]["accuracy"]` |  |
| `{{FIG:m.b31.1120.S.score.logloss}}` | `report.json` | `report.json["b31/1120"].metrics["S.score\|report"]["score"]["logloss"]` |  |
| `{{FIG:m.b31.1120.S.score.ece}}` | `report.json` | `report.json["b31/1120"].metrics["S.score\|report"]["score"]["ece"]` |  |
| `{{FIG:n.S.qimg}}` | `report.json` | `["b12/280"].metrics["S.qimg\|report"][<kind>]["n"]` | n is the same for every model (the same items) |
| `{{FIG:m.b12.280.S.qimg.accuracy}}` | `report.json` | `report.json["b12/280"].metrics["S.qimg\|report"]["qimg"]["accuracy"]` |  |
| `{{FIG:m.b12.280.S.qimg.logloss}}` | `report.json` | `report.json["b12/280"].metrics["S.qimg\|report"]["qimg"]["logloss"]` |  |
| `{{FIG:m.b12.280.S.qimg.ece}}` | `report.json` | `report.json["b12/280"].metrics["S.qimg\|report"]["qimg"]["ece"]` |  |
| `{{FIG:m.b12.1120.S.qimg.accuracy}}` | `report.json` | `report.json["b12/1120"].metrics["S.qimg\|report"]["qimg"]["accuracy"]` |  |
| `{{FIG:m.b12.1120.S.qimg.logloss}}` | `report.json` | `report.json["b12/1120"].metrics["S.qimg\|report"]["qimg"]["logloss"]` |  |
| `{{FIG:m.b12.1120.S.qimg.ece}}` | `report.json` | `report.json["b12/1120"].metrics["S.qimg\|report"]["qimg"]["ece"]` |  |
| `{{FIG:m.m16.sel.S.qimg.accuracy}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["S.qimg\|report"]["qimg"]["accuracy"]` |  |
| `{{FIG:m.m16.sel.S.qimg.logloss}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["S.qimg\|report"]["qimg"]["logloss"]` |  |
| `{{FIG:m.m16.sel.S.qimg.ece}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["S.qimg\|report"]["qimg"]["ece"]` |  |
| `{{FIG:m.b31.280.S.qimg.accuracy}}` | `report.json` | `report.json["b31/280"].metrics["S.qimg\|report"]["qimg"]["accuracy"]` |  |
| `{{FIG:m.b31.280.S.qimg.logloss}}` | `report.json` | `report.json["b31/280"].metrics["S.qimg\|report"]["qimg"]["logloss"]` |  |
| `{{FIG:m.b31.280.S.qimg.ece}}` | `report.json` | `report.json["b31/280"].metrics["S.qimg\|report"]["qimg"]["ece"]` |  |
| `{{FIG:m.b31.1120.S.qimg.accuracy}}` | `report.json` | `report.json["b31/1120"].metrics["S.qimg\|report"]["qimg"]["accuracy"]` |  |
| `{{FIG:m.b31.1120.S.qimg.logloss}}` | `report.json` | `report.json["b31/1120"].metrics["S.qimg\|report"]["qimg"]["logloss"]` |  |
| `{{FIG:m.b31.1120.S.qimg.ece}}` | `report.json` | `report.json["b31/1120"].metrics["S.qimg\|report"]["qimg"]["ece"]` |  |
| `{{FIG:n.R.noul}}` | `report.json` | `["b12/280"].metrics["R.noul\|report"][<kind>]["n"]` | n is the same for every model (the same items) |
| `{{FIG:m.b12.280.R.noul.accuracy}}` | `report.json` | `report.json["b12/280"].metrics["R.noul\|report"]["noul"]["accuracy"]` |  |
| `{{FIG:m.b12.280.R.noul.logloss}}` | `report.json` | `report.json["b12/280"].metrics["R.noul\|report"]["noul"]["logloss"]` |  |
| `{{FIG:m.b12.280.R.noul.ece}}` | `report.json` | `report.json["b12/280"].metrics["R.noul\|report"]["noul"]["ece"]` |  |
| `{{FIG:m.b12.1120.R.noul.accuracy}}` | `report.json` | `report.json["b12/1120"].metrics["R.noul\|report"]["noul"]["accuracy"]` |  |
| `{{FIG:m.b12.1120.R.noul.logloss}}` | `report.json` | `report.json["b12/1120"].metrics["R.noul\|report"]["noul"]["logloss"]` |  |
| `{{FIG:m.b12.1120.R.noul.ece}}` | `report.json` | `report.json["b12/1120"].metrics["R.noul\|report"]["noul"]["ece"]` |  |
| `{{FIG:m.m16.sel.R.noul.accuracy}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["R.noul\|report"]["noul"]["accuracy"]` |  |
| `{{FIG:m.m16.sel.R.noul.logloss}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["R.noul\|report"]["noul"]["logloss"]` |  |
| `{{FIG:m.m16.sel.R.noul.ece}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["R.noul\|report"]["noul"]["ece"]` |  |
| `{{FIG:m.b31.280.R.noul.accuracy}}` | `report.json` | `report.json["b31/280"].metrics["R.noul\|report"]["noul"]["accuracy"]` |  |
| `{{FIG:m.b31.280.R.noul.logloss}}` | `report.json` | `report.json["b31/280"].metrics["R.noul\|report"]["noul"]["logloss"]` |  |
| `{{FIG:m.b31.280.R.noul.ece}}` | `report.json` | `report.json["b31/280"].metrics["R.noul\|report"]["noul"]["ece"]` |  |
| `{{FIG:m.b31.1120.R.noul.accuracy}}` | `report.json` | `report.json["b31/1120"].metrics["R.noul\|report"]["noul"]["accuracy"]` |  |
| `{{FIG:m.b31.1120.R.noul.logloss}}` | `report.json` | `report.json["b31/1120"].metrics["R.noul\|report"]["noul"]["logloss"]` |  |
| `{{FIG:m.b31.1120.R.noul.ece}}` | `report.json` | `report.json["b31/1120"].metrics["R.noul\|report"]["noul"]["ece"]` |  |
| `{{FIG:n.R.choice}}` | `report.json` | `["b12/280"].metrics["R.choice\|report"][<kind>]["n"]` | n is the same for every model (the same items) |
| `{{FIG:m.b12.280.R.choice.accuracy}}` | `report.json` | `report.json["b12/280"].metrics["R.choice\|report"]["choice"]["accuracy"]` |  |
| `{{FIG:m.b12.280.R.choice.logloss}}` | `report.json` | `report.json["b12/280"].metrics["R.choice\|report"]["choice"]["logloss"]` |  |
| `{{FIG:m.b12.280.R.choice.ece}}` | `report.json` | `report.json["b12/280"].metrics["R.choice\|report"]["choice"]["ece"]` |  |
| `{{FIG:m.b12.1120.R.choice.accuracy}}` | `report.json` | `report.json["b12/1120"].metrics["R.choice\|report"]["choice"]["accuracy"]` |  |
| `{{FIG:m.b12.1120.R.choice.logloss}}` | `report.json` | `report.json["b12/1120"].metrics["R.choice\|report"]["choice"]["logloss"]` |  |
| `{{FIG:m.b12.1120.R.choice.ece}}` | `report.json` | `report.json["b12/1120"].metrics["R.choice\|report"]["choice"]["ece"]` |  |
| `{{FIG:m.m16.sel.R.choice.accuracy}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["R.choice\|report"]["choice"]["accuracy"]` |  |
| `{{FIG:m.m16.sel.R.choice.logloss}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["R.choice\|report"]["choice"]["logloss"]` |  |
| `{{FIG:m.m16.sel.R.choice.ece}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["R.choice\|report"]["choice"]["ece"]` |  |
| `{{FIG:m.b31.280.R.choice.accuracy}}` | `report.json` | `report.json["b31/280"].metrics["R.choice\|report"]["choice"]["accuracy"]` |  |
| `{{FIG:m.b31.280.R.choice.logloss}}` | `report.json` | `report.json["b31/280"].metrics["R.choice\|report"]["choice"]["logloss"]` |  |
| `{{FIG:m.b31.280.R.choice.ece}}` | `report.json` | `report.json["b31/280"].metrics["R.choice\|report"]["choice"]["ece"]` |  |
| `{{FIG:m.b31.1120.R.choice.accuracy}}` | `report.json` | `report.json["b31/1120"].metrics["R.choice\|report"]["choice"]["accuracy"]` |  |
| `{{FIG:m.b31.1120.R.choice.logloss}}` | `report.json` | `report.json["b31/1120"].metrics["R.choice\|report"]["choice"]["logloss"]` |  |
| `{{FIG:m.b31.1120.R.choice.ece}}` | `report.json` | `report.json["b31/1120"].metrics["R.choice\|report"]["choice"]["ece"]` |  |
| `{{FIG:n.I.noul}}` | `report.json` | `["b12/280"].metrics["I.noul\|report"][<kind>]["n"]` | n is the same for every model (the same items) |
| `{{FIG:m.b12.280.I.noul.accuracy}}` | `report.json` | `report.json["b12/280"].metrics["I.noul\|report"]["noul"]["accuracy"]` |  |
| `{{FIG:m.b12.280.I.noul.logloss}}` | `report.json` | `report.json["b12/280"].metrics["I.noul\|report"]["noul"]["logloss"]` |  |
| `{{FIG:m.b12.280.I.noul.ece}}` | `report.json` | `report.json["b12/280"].metrics["I.noul\|report"]["noul"]["ece"]` |  |
| `{{FIG:m.b12.1120.I.noul.accuracy}}` | `report.json` | `report.json["b12/1120"].metrics["I.noul\|report"]["noul"]["accuracy"]` |  |
| `{{FIG:m.b12.1120.I.noul.logloss}}` | `report.json` | `report.json["b12/1120"].metrics["I.noul\|report"]["noul"]["logloss"]` |  |
| `{{FIG:m.b12.1120.I.noul.ece}}` | `report.json` | `report.json["b12/1120"].metrics["I.noul\|report"]["noul"]["ece"]` |  |
| `{{FIG:m.m16.sel.I.noul.accuracy}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["I.noul\|report"]["noul"]["accuracy"]` |  |
| `{{FIG:m.m16.sel.I.noul.logloss}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["I.noul\|report"]["noul"]["logloss"]` |  |
| `{{FIG:m.m16.sel.I.noul.ece}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["I.noul\|report"]["noul"]["ece"]` |  |
| `{{FIG:m.b31.280.I.noul.accuracy}}` | `report.json` | `report.json["b31/280"].metrics["I.noul\|report"]["noul"]["accuracy"]` |  |
| `{{FIG:m.b31.280.I.noul.logloss}}` | `report.json` | `report.json["b31/280"].metrics["I.noul\|report"]["noul"]["logloss"]` |  |
| `{{FIG:m.b31.280.I.noul.ece}}` | `report.json` | `report.json["b31/280"].metrics["I.noul\|report"]["noul"]["ece"]` |  |
| `{{FIG:m.b31.1120.I.noul.accuracy}}` | `report.json` | `report.json["b31/1120"].metrics["I.noul\|report"]["noul"]["accuracy"]` |  |
| `{{FIG:m.b31.1120.I.noul.logloss}}` | `report.json` | `report.json["b31/1120"].metrics["I.noul\|report"]["noul"]["logloss"]` |  |
| `{{FIG:m.b31.1120.I.noul.ece}}` | `report.json` | `report.json["b31/1120"].metrics["I.noul\|report"]["noul"]["ece"]` |  |
| `{{FIG:n.I.choice}}` | `report.json` | `["b12/280"].metrics["I.choice\|report"][<kind>]["n"]` | n is the same for every model (the same items) |
| `{{FIG:m.b12.280.I.choice.accuracy}}` | `report.json` | `report.json["b12/280"].metrics["I.choice\|report"]["choice"]["accuracy"]` |  |
| `{{FIG:m.b12.280.I.choice.logloss}}` | `report.json` | `report.json["b12/280"].metrics["I.choice\|report"]["choice"]["logloss"]` |  |
| `{{FIG:m.b12.280.I.choice.ece}}` | `report.json` | `report.json["b12/280"].metrics["I.choice\|report"]["choice"]["ece"]` |  |
| `{{FIG:m.b12.1120.I.choice.accuracy}}` | `report.json` | `report.json["b12/1120"].metrics["I.choice\|report"]["choice"]["accuracy"]` |  |
| `{{FIG:m.b12.1120.I.choice.logloss}}` | `report.json` | `report.json["b12/1120"].metrics["I.choice\|report"]["choice"]["logloss"]` |  |
| `{{FIG:m.b12.1120.I.choice.ece}}` | `report.json` | `report.json["b12/1120"].metrics["I.choice\|report"]["choice"]["ece"]` |  |
| `{{FIG:m.m16.sel.I.choice.accuracy}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["I.choice\|report"]["choice"]["accuracy"]` |  |
| `{{FIG:m.m16.sel.I.choice.logloss}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["I.choice\|report"]["choice"]["logloss"]` |  |
| `{{FIG:m.m16.sel.I.choice.ece}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["I.choice\|report"]["choice"]["ece"]` |  |
| `{{FIG:m.b31.280.I.choice.accuracy}}` | `report.json` | `report.json["b31/280"].metrics["I.choice\|report"]["choice"]["accuracy"]` |  |
| `{{FIG:m.b31.280.I.choice.logloss}}` | `report.json` | `report.json["b31/280"].metrics["I.choice\|report"]["choice"]["logloss"]` |  |
| `{{FIG:m.b31.280.I.choice.ece}}` | `report.json` | `report.json["b31/280"].metrics["I.choice\|report"]["choice"]["ece"]` |  |
| `{{FIG:m.b31.1120.I.choice.accuracy}}` | `report.json` | `report.json["b31/1120"].metrics["I.choice\|report"]["choice"]["accuracy"]` |  |
| `{{FIG:m.b31.1120.I.choice.logloss}}` | `report.json` | `report.json["b31/1120"].metrics["I.choice\|report"]["choice"]["logloss"]` |  |
| `{{FIG:m.b31.1120.I.choice.ece}}` | `report.json` | `report.json["b31/1120"].metrics["I.choice\|report"]["choice"]["ece"]` |  |
| `{{FIG:n.I.score}}` | `report.json` | `["b12/280"].metrics["I.score\|report"][<kind>]["n"]` | n is the same for every model (the same items) |
| `{{FIG:m.b12.280.I.score.accuracy}}` | `report.json` | `report.json["b12/280"].metrics["I.score\|report"]["score"]["accuracy"]` |  |
| `{{FIG:m.b12.280.I.score.logloss}}` | `report.json` | `report.json["b12/280"].metrics["I.score\|report"]["score"]["logloss"]` |  |
| `{{FIG:m.b12.280.I.score.ece}}` | `report.json` | `report.json["b12/280"].metrics["I.score\|report"]["score"]["ece"]` |  |
| `{{FIG:m.b12.1120.I.score.accuracy}}` | `report.json` | `report.json["b12/1120"].metrics["I.score\|report"]["score"]["accuracy"]` |  |
| `{{FIG:m.b12.1120.I.score.logloss}}` | `report.json` | `report.json["b12/1120"].metrics["I.score\|report"]["score"]["logloss"]` |  |
| `{{FIG:m.b12.1120.I.score.ece}}` | `report.json` | `report.json["b12/1120"].metrics["I.score\|report"]["score"]["ece"]` |  |
| `{{FIG:m.m16.sel.I.score.accuracy}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["I.score\|report"]["score"]["accuracy"]` |  |
| `{{FIG:m.m16.sel.I.score.logloss}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["I.score\|report"]["score"]["logloss"]` |  |
| `{{FIG:m.m16.sel.I.score.ece}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["I.score\|report"]["score"]["ece"]` |  |
| `{{FIG:m.b31.280.I.score.accuracy}}` | `report.json` | `report.json["b31/280"].metrics["I.score\|report"]["score"]["accuracy"]` |  |
| `{{FIG:m.b31.280.I.score.logloss}}` | `report.json` | `report.json["b31/280"].metrics["I.score\|report"]["score"]["logloss"]` |  |
| `{{FIG:m.b31.280.I.score.ece}}` | `report.json` | `report.json["b31/280"].metrics["I.score\|report"]["score"]["ece"]` |  |
| `{{FIG:m.b31.1120.I.score.accuracy}}` | `report.json` | `report.json["b31/1120"].metrics["I.score\|report"]["score"]["accuracy"]` |  |
| `{{FIG:m.b31.1120.I.score.logloss}}` | `report.json` | `report.json["b31/1120"].metrics["I.score\|report"]["score"]["logloss"]` |  |
| `{{FIG:m.b31.1120.I.score.ece}}` | `report.json` | `report.json["b31/1120"].metrics["I.score\|report"]["score"]["ece"]` |  |
| `{{FIG:m.b12.280.S.score.mae}}` | `report.json` | `report.json["b12/280"].metrics["S.score\|report"]["score"]["mae"]` |  |
| `{{FIG:n_present.S.box}}` | `report.json` | `["b12/280"].metrics["S.box\|report"]["box"]["n_present"]` |  |
| `{{FIG:m.b12.280.S.box.mean_iou}}` | `report.json` | `report.json["b12/280"].metrics["S.box\|report"]["box"]["mean_iou"]` |  |
| `{{FIG:m.b12.1120.S.box.mean_iou}}` | `report.json` | `report.json["b12/1120"].metrics["S.box\|report"]["box"]["mean_iou"]` |  |
| `{{FIG:m.m16.sel.S.box.mean_iou}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["S.box\|report"]["box"]["mean_iou"]` |  |
| `{{FIG:m.b31.280.S.box.mean_iou}}` | `report.json` | `report.json["b31/280"].metrics["S.box\|report"]["box"]["mean_iou"]` |  |
| `{{FIG:m.b31.1120.S.box.mean_iou}}` | `report.json` | `report.json["b31/1120"].metrics["S.box\|report"]["box"]["mean_iou"]` |  |
| `{{FIG:m.b12.280.S.box.iou50}}` | `report.json` | `report.json["b12/280"].metrics["S.box\|report"]["box"]["iou50"]` |  |
| `{{FIG:m.b12.1120.S.box.iou50}}` | `report.json` | `report.json["b12/1120"].metrics["S.box\|report"]["box"]["iou50"]` |  |
| `{{FIG:m.m16.sel.S.box.iou50}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["S.box\|report"]["box"]["iou50"]` |  |
| `{{FIG:m.b31.280.S.box.iou50}}` | `report.json` | `report.json["b31/280"].metrics["S.box\|report"]["box"]["iou50"]` |  |
| `{{FIG:m.b31.1120.S.box.iou50}}` | `report.json` | `report.json["b31/1120"].metrics["S.box\|report"]["box"]["iou50"]` |  |
| `{{FIG:m.b12.280.S.box.iou75}}` | `report.json` | `report.json["b12/280"].metrics["S.box\|report"]["box"]["iou75"]` |  |
| `{{FIG:m.b12.1120.S.box.iou75}}` | `report.json` | `report.json["b12/1120"].metrics["S.box\|report"]["box"]["iou75"]` |  |
| `{{FIG:m.m16.sel.S.box.iou75}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["S.box\|report"]["box"]["iou75"]` |  |
| `{{FIG:m.b31.280.S.box.iou75}}` | `report.json` | `report.json["b31/280"].metrics["S.box\|report"]["box"]["iou75"]` |  |
| `{{FIG:m.b31.1120.S.box.iou75}}` | `report.json` | `report.json["b31/1120"].metrics["S.box\|report"]["box"]["iou75"]` |  |
| `{{FIG:m.b12.280.S.box.edge_mae}}` | `report.json` | `report.json["b12/280"].metrics["S.box\|report"]["box"]["edge_mae"]` |  |
| `{{FIG:m.b12.1120.S.box.edge_mae}}` | `report.json` | `report.json["b12/1120"].metrics["S.box\|report"]["box"]["edge_mae"]` |  |
| `{{FIG:m.m16.sel.S.box.edge_mae}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["S.box\|report"]["box"]["edge_mae"]` |  |
| `{{FIG:m.b31.280.S.box.edge_mae}}` | `report.json` | `report.json["b31/280"].metrics["S.box\|report"]["box"]["edge_mae"]` |  |
| `{{FIG:m.b31.1120.S.box.edge_mae}}` | `report.json` | `report.json["b31/1120"].metrics["S.box\|report"]["box"]["edge_mae"]` |  |
| `{{FIG:m.b12.280.S.box.conf_ece}}` | `report.json` | `report.json["b12/280"].metrics["S.box\|report"]["box"]["conf_ece"]` |  |
| `{{FIG:m.b12.1120.S.box.conf_ece}}` | `report.json` | `report.json["b12/1120"].metrics["S.box\|report"]["box"]["conf_ece"]` |  |
| `{{FIG:m.m16.sel.S.box.conf_ece}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["S.box\|report"]["box"]["conf_ece"]` |  |
| `{{FIG:m.b31.280.S.box.conf_ece}}` | `report.json` | `report.json["b31/280"].metrics["S.box\|report"]["box"]["conf_ece"]` |  |
| `{{FIG:m.b31.1120.S.box.conf_ece}}` | `report.json` | `report.json["b31/1120"].metrics["S.box\|report"]["box"]["conf_ece"]` |  |
| `{{FIG:n_present.R.box}}` | `report.json` | `["b12/280"].metrics["R.box\|report"]["box"]["n_present"]` |  |
| `{{FIG:m.b12.280.R.box.mean_iou}}` | `report.json` | `report.json["b12/280"].metrics["R.box\|report"]["box"]["mean_iou"]` |  |
| `{{FIG:m.b12.1120.R.box.mean_iou}}` | `report.json` | `report.json["b12/1120"].metrics["R.box\|report"]["box"]["mean_iou"]` |  |
| `{{FIG:m.m16.sel.R.box.mean_iou}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["R.box\|report"]["box"]["mean_iou"]` |  |
| `{{FIG:m.b31.280.R.box.mean_iou}}` | `report.json` | `report.json["b31/280"].metrics["R.box\|report"]["box"]["mean_iou"]` |  |
| `{{FIG:m.b31.1120.R.box.mean_iou}}` | `report.json` | `report.json["b31/1120"].metrics["R.box\|report"]["box"]["mean_iou"]` |  |
| `{{FIG:m.b12.280.R.box.iou50}}` | `report.json` | `report.json["b12/280"].metrics["R.box\|report"]["box"]["iou50"]` |  |
| `{{FIG:m.b12.1120.R.box.iou50}}` | `report.json` | `report.json["b12/1120"].metrics["R.box\|report"]["box"]["iou50"]` |  |
| `{{FIG:m.m16.sel.R.box.iou50}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["R.box\|report"]["box"]["iou50"]` |  |
| `{{FIG:m.b31.280.R.box.iou50}}` | `report.json` | `report.json["b31/280"].metrics["R.box\|report"]["box"]["iou50"]` |  |
| `{{FIG:m.b31.1120.R.box.iou50}}` | `report.json` | `report.json["b31/1120"].metrics["R.box\|report"]["box"]["iou50"]` |  |
| `{{FIG:m.b12.280.R.box.iou75}}` | `report.json` | `report.json["b12/280"].metrics["R.box\|report"]["box"]["iou75"]` |  |
| `{{FIG:m.b12.1120.R.box.iou75}}` | `report.json` | `report.json["b12/1120"].metrics["R.box\|report"]["box"]["iou75"]` |  |
| `{{FIG:m.m16.sel.R.box.iou75}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["R.box\|report"]["box"]["iou75"]` |  |
| `{{FIG:m.b31.280.R.box.iou75}}` | `report.json` | `report.json["b31/280"].metrics["R.box\|report"]["box"]["iou75"]` |  |
| `{{FIG:m.b31.1120.R.box.iou75}}` | `report.json` | `report.json["b31/1120"].metrics["R.box\|report"]["box"]["iou75"]` |  |
| `{{FIG:m.b12.280.R.box.edge_mae}}` | `report.json` | `report.json["b12/280"].metrics["R.box\|report"]["box"]["edge_mae"]` |  |
| `{{FIG:m.b12.1120.R.box.edge_mae}}` | `report.json` | `report.json["b12/1120"].metrics["R.box\|report"]["box"]["edge_mae"]` |  |
| `{{FIG:m.m16.sel.R.box.edge_mae}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["R.box\|report"]["box"]["edge_mae"]` |  |
| `{{FIG:m.b31.280.R.box.edge_mae}}` | `report.json` | `report.json["b31/280"].metrics["R.box\|report"]["box"]["edge_mae"]` |  |
| `{{FIG:m.b31.1120.R.box.edge_mae}}` | `report.json` | `report.json["b31/1120"].metrics["R.box\|report"]["box"]["edge_mae"]` |  |
| `{{FIG:m.b12.280.R.box.conf_ece}}` | `report.json` | `report.json["b12/280"].metrics["R.box\|report"]["box"]["conf_ece"]` |  |
| `{{FIG:m.b12.1120.R.box.conf_ece}}` | `report.json` | `report.json["b12/1120"].metrics["R.box\|report"]["box"]["conf_ece"]` |  |
| `{{FIG:m.m16.sel.R.box.conf_ece}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["R.box\|report"]["box"]["conf_ece"]` |  |
| `{{FIG:m.b31.280.R.box.conf_ece}}` | `report.json` | `report.json["b31/280"].metrics["R.box\|report"]["box"]["conf_ece"]` |  |
| `{{FIG:m.b31.1120.R.box.conf_ece}}` | `report.json` | `report.json["b31/1120"].metrics["R.box\|report"]["box"]["conf_ece"]` |  |
| `{{FIG:n_absent.S.box}}` | `report.json` | `["b12/280"].metrics["S.box\|report"]["box"]["n_absent"]` |  |
| `{{FIG:m.b12.280.S.box.absent_auroc}}` | `report.json` | `report.json["b12/280"].metrics["S.box\|report"]["box"]["absent_auroc"]` |  |
| `{{FIG:m.b12.1120.S.box.absent_auroc}}` | `report.json` | `report.json["b12/1120"].metrics["S.box\|report"]["box"]["absent_auroc"]` |  |
| `{{FIG:m.m16.sel.S.box.absent_auroc}}` | `report.json` | `report.json["m16/<M16 start budget>"].metrics["S.box\|report"]["box"]["absent_auroc"]` |  |
| `{{FIG:m.b31.280.S.box.absent_auroc}}` | `report.json` | `report.json["b31/280"].metrics["S.box\|report"]["box"]["absent_auroc"]` |  |
| `{{FIG:m.b31.1120.S.box.absent_auroc}}` | `report.json` | `report.json["b31/1120"].metrics["S.box\|report"]["box"]["absent_auroc"]` |  |
| `{{FIG:box_companion_auroc.b12.280}}` | `GAP: not in report.py` | `AUROC of the noul answer over present and absent S Box items` | the report scores the Box confidence only; the companion noul's AUROC needs a paired noul per Box item |
| `{{FIG:box_companion_auroc.b12.1120}}` | `GAP: not in report.py` | `AUROC of the noul answer over present and absent S Box items` | the report scores the Box confidence only; the companion noul's AUROC needs a paired noul per Box item |
| `{{FIG:box_companion_auroc.m16.sel}}` | `GAP: not in report.py` | `AUROC of the noul answer over present and absent S Box items` | the report scores the Box confidence only; the companion noul's AUROC needs a paired noul per Box item |
| `{{FIG:box_companion_auroc.b31.280}}` | `GAP: not in report.py` | `AUROC of the noul answer over present and absent S Box items` | the report scores the Box confidence only; the companion noul's AUROC needs a paired noul per Box item |
| `{{FIG:box_companion_auroc.b31.1120}}` | `GAP: not in report.py` | `AUROC of the noul answer over present and absent S Box items` | the report scores the Box confidence only; the companion noul's AUROC needs a paired noul per Box item |
| `{{FIG:box.confidence_map}}` | `(box_confidence.json of each model, fitted by the run)` | `isotonic fit, calibration half, n points` | TODO: ship it, see records_slots.md |
| `{{FIG:pair.n.S_accuracy}}` | `report.json` | `report.json["m16-b12/<budget>"]["S\|report\|accuracy"]["n"]` |  |
| `{{FIG:pair.diff.S_accuracy}}` | `report.json` | `report.json["m16-b12/<budget>"]["S\|report\|accuracy"]["diff"]` |  |
| `{{FIG:pair.lo.S_accuracy}}` | `report.json` | `report.json["m16-b12/<budget>"]["S\|report\|accuracy"]["lo"]` |  |
| `{{FIG:pair.hi.S_accuracy}}` | `report.json` | `report.json["m16-b12/<budget>"]["S\|report\|accuracy"]["hi"]` |  |
| `{{FIG:pair.n.R_accuracy}}` | `report.json` | `report.json["m16-b12/<budget>"]["R\|report\|accuracy"]["n"]` |  |
| `{{FIG:pair.diff.R_accuracy}}` | `report.json` | `report.json["m16-b12/<budget>"]["R\|report\|accuracy"]["diff"]` |  |
| `{{FIG:pair.lo.R_accuracy}}` | `report.json` | `report.json["m16-b12/<budget>"]["R\|report\|accuracy"]["lo"]` |  |
| `{{FIG:pair.hi.R_accuracy}}` | `report.json` | `report.json["m16-b12/<budget>"]["R\|report\|accuracy"]["hi"]` |  |
| `{{FIG:pair.n.I_accuracy}}` | `report.json` | `report.json["m16-b12/<budget>"]["I\|report\|accuracy"]["n"]` |  |
| `{{FIG:pair.diff.I_accuracy}}` | `report.json` | `report.json["m16-b12/<budget>"]["I\|report\|accuracy"]["diff"]` |  |
| `{{FIG:pair.lo.I_accuracy}}` | `report.json` | `report.json["m16-b12/<budget>"]["I\|report\|accuracy"]["lo"]` |  |
| `{{FIG:pair.hi.I_accuracy}}` | `report.json` | `report.json["m16-b12/<budget>"]["I\|report\|accuracy"]["hi"]` |  |
| `{{FIG:pair.n.S_logloss}}` | `report.json` | `report.json["m16-b12/<budget>"]["S\|report\|logloss"]["n"]` |  |
| `{{FIG:pair.diff.S_logloss}}` | `report.json` | `report.json["m16-b12/<budget>"]["S\|report\|logloss"]["diff"]` |  |
| `{{FIG:pair.lo.S_logloss}}` | `report.json` | `report.json["m16-b12/<budget>"]["S\|report\|logloss"]["lo"]` |  |
| `{{FIG:pair.hi.S_logloss}}` | `report.json` | `report.json["m16-b12/<budget>"]["S\|report\|logloss"]["hi"]` |  |
| `{{FIG:pair.n.R_logloss}}` | `report.json` | `report.json["m16-b12/<budget>"]["R\|report\|logloss"]["n"]` |  |
| `{{FIG:pair.diff.R_logloss}}` | `report.json` | `report.json["m16-b12/<budget>"]["R\|report\|logloss"]["diff"]` |  |
| `{{FIG:pair.lo.R_logloss}}` | `report.json` | `report.json["m16-b12/<budget>"]["R\|report\|logloss"]["lo"]` |  |
| `{{FIG:pair.hi.R_logloss}}` | `report.json` | `report.json["m16-b12/<budget>"]["R\|report\|logloss"]["hi"]` |  |
| `{{FIG:pair.n.I_logloss}}` | `report.json` | `report.json["m16-b12/<budget>"]["I\|report\|logloss"]["n"]` |  |
| `{{FIG:pair.diff.I_logloss}}` | `report.json` | `report.json["m16-b12/<budget>"]["I\|report\|logloss"]["diff"]` |  |
| `{{FIG:pair.lo.I_logloss}}` | `report.json` | `report.json["m16-b12/<budget>"]["I\|report\|logloss"]["lo"]` |  |
| `{{FIG:pair.hi.I_logloss}}` | `report.json` | `report.json["m16-b12/<budget>"]["I\|report\|logloss"]["hi"]` |  |
| `{{FIG:pair.n.S_mean_iou}}` | `report.json` | `report.json["m16-b12/<budget>"]["S\|report\|mean_iou"]["n"]` |  |
| `{{FIG:pair.diff.S_mean_iou}}` | `report.json` | `report.json["m16-b12/<budget>"]["S\|report\|mean_iou"]["diff"]` |  |
| `{{FIG:pair.lo.S_mean_iou}}` | `report.json` | `report.json["m16-b12/<budget>"]["S\|report\|mean_iou"]["lo"]` |  |
| `{{FIG:pair.hi.S_mean_iou}}` | `report.json` | `report.json["m16-b12/<budget>"]["S\|report\|mean_iou"]["hi"]` |  |
| `{{FIG:pair.n.R_mean_iou}}` | `report.json` | `report.json["m16-b12/<budget>"]["R\|report\|mean_iou"]["n"]` |  |
| `{{FIG:pair.diff.R_mean_iou}}` | `report.json` | `report.json["m16-b12/<budget>"]["R\|report\|mean_iou"]["diff"]` |  |
| `{{FIG:pair.lo.R_mean_iou}}` | `report.json` | `report.json["m16-b12/<budget>"]["R\|report\|mean_iou"]["lo"]` |  |
| `{{FIG:pair.hi.R_mean_iou}}` | `report.json` | `report.json["m16-b12/<budget>"]["R\|report\|mean_iou"]["hi"]` |  |
| `{{FIG:pair.n.S_iou50}}` | `report.json` | `report.json["m16-b12/<budget>"]["S\|report\|iou50"]["n"]` |  |
| `{{FIG:pair.diff.S_iou50}}` | `report.json` | `report.json["m16-b12/<budget>"]["S\|report\|iou50"]["diff"]` |  |
| `{{FIG:pair.lo.S_iou50}}` | `report.json` | `report.json["m16-b12/<budget>"]["S\|report\|iou50"]["lo"]` |  |
| `{{FIG:pair.hi.S_iou50}}` | `report.json` | `report.json["m16-b12/<budget>"]["S\|report\|iou50"]["hi"]` |  |
| `{{FIG:pair.n.R_iou50}}` | `report.json` | `report.json["m16-b12/<budget>"]["R\|report\|iou50"]["n"]` |  |
| `{{FIG:pair.diff.R_iou50}}` | `report.json` | `report.json["m16-b12/<budget>"]["R\|report\|iou50"]["diff"]` |  |
| `{{FIG:pair.lo.R_iou50}}` | `report.json` | `report.json["m16-b12/<budget>"]["R\|report\|iou50"]["lo"]` |  |
| `{{FIG:pair.hi.R_iou50}}` | `report.json` | `report.json["m16-b12/<budget>"]["R\|report\|iou50"]["hi"]` |  |
| `{{FIG:pair.gate_verdict}}` | `REPORT.md section 2` | `PASS/FAIL columns, point and lower end` | verdict text: the point rule and the strict (interval lower end) rule |
| `{{FIG:corner.b12.280}}` | `results/m/b12/on280/corner.json` | `passed (of 8)` |  |
| `{{FIG:corner.b12.1120}}` | `results/m/b12/on1120/corner.json` | `passed (of 8)` |  |
| `{{FIG:corner.m16.sel}}` | `results/m/m16/onsel/corner.json` | `passed (of 8)` |  |
| `{{FIG:corner.b31.280}}` | `results/m/b31/on280/corner.json` | `passed (of 8)` |  |
| `{{FIG:corner.b31.1120}}` | `results/m/b31/on1120/corner.json` | `passed (of 8)` |  |
| `{{FIG:decision.default_budget}}` | `(decision of the maintainers after the run)` | `` | TODO: default image budget in families.py or server flags; and the refuse-at-start rule |
| `{{FIG:parity.b12.max_dp}}` | `report.json` | `report.json["parity/b12/on280"]["max_dp"]` |  |
| `{{FIG:parity.b12.changed}}` | `report.json` | `report.json["parity/b12/on280"]["choices_changed"]` |  |
| `{{FIG:parity.b12.rep_max_dp}}` | `report.json` | `report.json["parity/b12/repeat"]["max_dp"]` |  |
| `{{FIG:parity.b12.rep_changed}}` | `report.json` | `report.json["parity/b12/repeat"]["choices_changed"]` |  |
| `{{FIG:parity.b12.verdict}}` | `REPORT.md section 5` | `verdict column` | not stored in report.json: the filler derives it by the stated rule |
| `{{FIG:parity.m16.max_dp}}` | `report.json` | `report.json["parity/m16/onsel"]["max_dp"]` |  |
| `{{FIG:parity.m16.changed}}` | `report.json` | `report.json["parity/m16/onsel"]["choices_changed"]` |  |
| `{{FIG:parity.m16.rep_max_dp}}` | `report.json` | `report.json["parity/m16/repeat"]["max_dp"]` |  |
| `{{FIG:parity.m16.rep_changed}}` | `report.json` | `report.json["parity/m16/repeat"]["choices_changed"]` |  |
| `{{FIG:parity.m16.verdict}}` | `REPORT.md section 5` | `verdict column` | not stored in report.json: the filler derives it by the stated rule |
| `{{FIG:parity.b31.max_dp}}` | `report.json` | `report.json["parity/b31/on280"]["max_dp"]` |  |
| `{{FIG:parity.b31.changed}}` | `report.json` | `report.json["parity/b31/on280"]["choices_changed"]` |  |
| `{{FIG:parity.b31.rep_max_dp}}` | `report.json` | `report.json["parity/b31/repeat"]["max_dp"]` |  |
| `{{FIG:parity.b31.rep_changed}}` | `report.json` | `report.json["parity/b31/repeat"]["choices_changed"]` |  |
| `{{FIG:parity.b31.verdict}}` | `REPORT.md section 5` | `verdict column` | not stored in report.json: the filler derives it by the stated rule |
| `{{FIG:start.b12.off1.ready}}` | `results/starts.jsonl` | `rc == 0` |  |
| `{{FIG:start.b12.off1.seconds}}` | `results/starts.jsonl` | `seconds_to_ready` |  |
| `{{FIG:start.b12.off1.mib}}` | `results/starts.jsonl` | `gpu_used_mib_after_ready` |  |
| `{{FIG:start.b12.off1.peak}}` | `results/starts.jsonl` | `start_window.peak_mib` |  |
| `{{FIG:start.b12.on280.ready}}` | `results/starts.jsonl` | `rc == 0` |  |
| `{{FIG:start.b12.on280.seconds}}` | `results/starts.jsonl` | `seconds_to_ready` |  |
| `{{FIG:start.b12.on280.mib}}` | `results/starts.jsonl` | `gpu_used_mib_after_ready` |  |
| `{{FIG:start.b12.on280.peak}}` | `results/starts.jsonl` | `start_window.peak_mib` |  |
| `{{FIG:start.b12.on1120.ready}}` | `results/starts.jsonl` | `rc == 0` |  |
| `{{FIG:start.b12.on1120.seconds}}` | `results/starts.jsonl` | `seconds_to_ready` |  |
| `{{FIG:start.b12.on1120.mib}}` | `results/starts.jsonl` | `gpu_used_mib_after_ready` |  |
| `{{FIG:start.b12.on1120.peak}}` | `results/starts.jsonl` | `start_window.peak_mib` |  |
| `{{FIG:start.b12.m128.ready}}` | `results/starts.jsonl` | `rc == 0` |  |
| `{{FIG:start.b12.m128.seconds}}` | `results/starts.jsonl` | `seconds_to_ready` |  |
| `{{FIG:start.b12.m128.mib}}` | `results/starts.jsonl` | `gpu_used_mib_after_ready` |  |
| `{{FIG:start.b12.m128.peak}}` | `results/starts.jsonl` | `start_window.peak_mib` |  |
| `{{FIG:start.m16.off1.ready}}` | `results/starts.jsonl` | `rc == 0` |  |
| `{{FIG:start.m16.off1.seconds}}` | `results/starts.jsonl` | `seconds_to_ready` |  |
| `{{FIG:start.m16.off1.mib}}` | `results/starts.jsonl` | `gpu_used_mib_after_ready` |  |
| `{{FIG:start.m16.off1.peak}}` | `results/starts.jsonl` | `start_window.peak_mib` |  |
| `{{FIG:start.m16.on.ready}}` | `results/starts.jsonl` | `rc == 0` |  |
| `{{FIG:start.m16.on.seconds}}` | `results/starts.jsonl` | `seconds_to_ready` |  |
| `{{FIG:start.m16.on.mib}}` | `results/starts.jsonl` | `gpu_used_mib_after_ready` |  |
| `{{FIG:start.m16.on.peak}}` | `results/starts.jsonl` | `start_window.peak_mib` |  |
| `{{FIG:start.m16.m128.ready}}` | `results/starts.jsonl` | `rc == 0` |  |
| `{{FIG:start.m16.m128.seconds}}` | `results/starts.jsonl` | `seconds_to_ready` |  |
| `{{FIG:start.m16.m128.mib}}` | `results/starts.jsonl` | `gpu_used_mib_after_ready` |  |
| `{{FIG:start.m16.m128.peak}}` | `results/starts.jsonl` | `start_window.peak_mib` |  |
| `{{FIG:start.b31.off1.ready}}` | `results/starts.jsonl` | `rc == 0` |  |
| `{{FIG:start.b31.off1.seconds}}` | `results/starts.jsonl` | `seconds_to_ready` |  |
| `{{FIG:start.b31.off1.mib}}` | `results/starts.jsonl` | `gpu_used_mib_after_ready` |  |
| `{{FIG:start.b31.off1.peak}}` | `results/starts.jsonl` | `start_window.peak_mib` |  |
| `{{FIG:start.b31.on280.ready}}` | `results/starts.jsonl` | `rc == 0` |  |
| `{{FIG:start.b31.on280.seconds}}` | `results/starts.jsonl` | `seconds_to_ready` |  |
| `{{FIG:start.b31.on280.mib}}` | `results/starts.jsonl` | `gpu_used_mib_after_ready` |  |
| `{{FIG:start.b31.on280.peak}}` | `results/starts.jsonl` | `start_window.peak_mib` |  |
| `{{FIG:start.b31.on1120.ready}}` | `results/starts.jsonl` | `rc == 0` |  |
| `{{FIG:start.b31.on1120.seconds}}` | `results/starts.jsonl` | `seconds_to_ready` |  |
| `{{FIG:start.b31.on1120.mib}}` | `results/starts.jsonl` | `gpu_used_mib_after_ready` |  |
| `{{FIG:start.b31.on1120.peak}}` | `results/starts.jsonl` | `start_window.peak_mib` |  |
| `{{FIG:start.b31.m128.ready}}` | `results/starts.jsonl` | `rc == 0` |  |
| `{{FIG:start.b31.m128.seconds}}` | `results/starts.jsonl` | `seconds_to_ready` |  |
| `{{FIG:start.b31.m128.mib}}` | `results/starts.jsonl` | `gpu_used_mib_after_ready` |  |
| `{{FIG:start.b31.m128.peak}}` | `results/starts.jsonl` | `start_window.peak_mib` |  |
| `{{FIG:lat.b12.280.1}}` | `results/m/b12/on280/latency.json` | `wall_ms p50 / p95 of arm images=1, status 200` | from latency_table.py columns |
| `{{FIG:lat.b12.280.4}}` | `results/m/b12/on280/latency.json` | `wall_ms p50 / p95 of arm images=4, status 200` | from latency_table.py columns |
| `{{FIG:lat.b12.280.16}}` | `results/m/b12/on280/latency.json` | `wall_ms p50 / p95 of arm images=16, status 200` | from latency_table.py columns |
| `{{FIG:lat.b12.280.tok}}` | `results/m/b12/on280/latency.json` | `tokens(4) minus tokens(1), divided by 3` |  |
| `{{FIG:lat.b12.1120.1}}` | `results/m/b12/on1120/latency.json` | `wall_ms p50 / p95 of arm images=1, status 200` | from latency_table.py columns |
| `{{FIG:lat.b12.1120.4}}` | `results/m/b12/on1120/latency.json` | `wall_ms p50 / p95 of arm images=4, status 200` | from latency_table.py columns |
| `{{FIG:lat.b12.1120.16}}` | `results/m/b12/on1120/latency.json` | `wall_ms p50 / p95 of arm images=16, status 200` | from latency_table.py columns |
| `{{FIG:lat.b12.1120.tok}}` | `results/m/b12/on1120/latency.json` | `tokens(4) minus tokens(1), divided by 3` |  |
| `{{FIG:lat.m16.sel.1}}` | `results/m/m16/on/latency.json` | `wall_ms p50 / p95 of arm images=1, status 200` | from latency_table.py columns |
| `{{FIG:lat.m16.sel.4}}` | `results/m/m16/on/latency.json` | `wall_ms p50 / p95 of arm images=4, status 200` | from latency_table.py columns |
| `{{FIG:lat.m16.sel.16}}` | `results/m/m16/on/latency.json` | `wall_ms p50 / p95 of arm images=16, status 200` | from latency_table.py columns |
| `{{FIG:lat.m16.sel.tok}}` | `results/m/m16/on/latency.json` | `tokens(4) minus tokens(1), divided by 3` |  |
| `{{FIG:lat.b31.280.1}}` | `results/m/b31/on280/latency.json` | `wall_ms p50 / p95 of arm images=1, status 200` | from latency_table.py columns |
| `{{FIG:lat.b31.280.4}}` | `results/m/b31/on280/latency.json` | `wall_ms p50 / p95 of arm images=4, status 200` | from latency_table.py columns |
| `{{FIG:lat.b31.280.16}}` | `results/m/b31/on280/latency.json` | `wall_ms p50 / p95 of arm images=16, status 200` | from latency_table.py columns |
| `{{FIG:lat.b31.280.tok}}` | `results/m/b31/on280/latency.json` | `tokens(4) minus tokens(1), divided by 3` |  |
| `{{FIG:lat.b31.1120.1}}` | `results/m/b31/on1120/latency.json` | `wall_ms p50 / p95 of arm images=1, status 200` | from latency_table.py columns |
| `{{FIG:lat.b31.1120.4}}` | `results/m/b31/on1120/latency.json` | `wall_ms p50 / p95 of arm images=4, status 200` | from latency_table.py columns |
| `{{FIG:lat.b31.1120.16}}` | `results/m/b31/on1120/latency.json` | `wall_ms p50 / p95 of arm images=16, status 200` | from latency_table.py columns |
| `{{FIG:lat.b31.1120.tok}}` | `results/m/b31/on1120/latency.json` | `tokens(4) minus tokens(1), divided by 3` |  |
| `{{FIG:lat.b12.280.box}}` | `results/m/b12/on280/lat2.json` | `box wall_ms p50 / p95` |  |
| `{{FIG:lat.b31.280.box}}` | `results/m/b31/on280/lat2.json` | `box wall_ms p50 / p95` |  |
| `{{FIG:prefix.b12.280}}` | `results/m/b12/on280/prefix.json` | `other_question_read_from_cache of n, max_dp_first_vs_again` | sentence-level summary |
| `{{FIG:fitT.b12.280.T}}` | `results/m/b12/on280/fitT.json` | `T` |  |
| `{{FIG:fitT.b12.280.cb}}` | `results/m/b12/on280/fitT.json` | `calib_logloss_before` |  |
| `{{FIG:fitT.b12.280.ca}}` | `results/m/b12/on280/fitT.json` | `calib_logloss_after` |  |
| `{{FIG:fitT.b12.280.rb}}` | `results/m/b12/on280/fitT.json` | `report_logloss_before` |  |
| `{{FIG:fitT.b12.280.ra}}` | `results/m/b12/on280/fitT.json` | `report_logloss_after` |  |
| `{{FIG:fitT.b12.1120.T}}` | `results/m/b12/on1120/fitT.json` | `T` |  |
| `{{FIG:fitT.b12.1120.cb}}` | `results/m/b12/on1120/fitT.json` | `calib_logloss_before` |  |
| `{{FIG:fitT.b12.1120.ca}}` | `results/m/b12/on1120/fitT.json` | `calib_logloss_after` |  |
| `{{FIG:fitT.b12.1120.rb}}` | `results/m/b12/on1120/fitT.json` | `report_logloss_before` |  |
| `{{FIG:fitT.b12.1120.ra}}` | `results/m/b12/on1120/fitT.json` | `report_logloss_after` |  |
| `{{FIG:fitT.m16.sel.T}}` | `results/m/m16/on/fitT.json` | `T` |  |
| `{{FIG:fitT.m16.sel.cb}}` | `results/m/m16/on/fitT.json` | `calib_logloss_before` |  |
| `{{FIG:fitT.m16.sel.ca}}` | `results/m/m16/on/fitT.json` | `calib_logloss_after` |  |
| `{{FIG:fitT.m16.sel.rb}}` | `results/m/m16/on/fitT.json` | `report_logloss_before` |  |
| `{{FIG:fitT.m16.sel.ra}}` | `results/m/m16/on/fitT.json` | `report_logloss_after` |  |
| `{{FIG:fitT.b31.280.T}}` | `results/m/b31/on280/fitT.json` | `T` |  |
| `{{FIG:fitT.b31.280.cb}}` | `results/m/b31/on280/fitT.json` | `calib_logloss_before` |  |
| `{{FIG:fitT.b31.280.ca}}` | `results/m/b31/on280/fitT.json` | `calib_logloss_after` |  |
| `{{FIG:fitT.b31.280.rb}}` | `results/m/b31/on280/fitT.json` | `report_logloss_before` |  |
| `{{FIG:fitT.b31.280.ra}}` | `results/m/b31/on280/fitT.json` | `report_logloss_after` |  |
| `{{FIG:fitT.b31.1120.T}}` | `results/m/b31/on1120/fitT.json` | `T` |  |
| `{{FIG:fitT.b31.1120.cb}}` | `results/m/b31/on1120/fitT.json` | `calib_logloss_before` |  |
| `{{FIG:fitT.b31.1120.ca}}` | `results/m/b31/on1120/fitT.json` | `calib_logloss_after` |  |
| `{{FIG:fitT.b31.1120.rb}}` | `results/m/b31/on1120/fitT.json` | `report_logloss_before` |  |
| `{{FIG:fitT.b31.1120.ra}}` | `results/m/b31/on1120/fitT.json` | `report_logloss_after` |  |
| `{{FIG:tree.fast.mean}}` | `results/tree_b12.json` | `fast_vs_exact.mean` |  |
| `{{FIG:tree.fast.p99}}` | `results/tree_b12.json` | `fast_vs_exact.p99` |  |
| `{{FIG:tree.fast.max}}` | `results/tree_b12.json` | `fast_vs_exact.max` |  |
| `{{FIG:tree.greedy.mean}}` | `results/tree_b12.json` | `greedy_vs_exact.mean` |  |
| `{{FIG:tree.greedy.p99}}` | `results/tree_b12.json` | `greedy_vs_exact.p99` |  |
| `{{FIG:tree.greedy.max}}` | `results/tree_b12.json` | `greedy_vs_exact.max` |  |
| `{{FIG:tree.complete}}` | `results/tree_b12.json` | `coordinates (trees with mass >= 0.99)` |  |
| `{{FIG:tree.capped}}` | `results/tree_b12.json` | `200 minus coordinates` | trees that hit the 64-decode cap |
| `{{FIG:q35.n.S_accuracy}}` | `report.json` | `report.json["q35-new-minus-legacy"]["S\|report\|accuracy"]["n"]` |  |
| `{{FIG:q35.diff.S_accuracy}}` | `report.json` | `report.json["q35-new-minus-legacy"]["S\|report\|accuracy"]["diff"]` |  |
| `{{FIG:q35.lo.S_accuracy}}` | `report.json` | `report.json["q35-new-minus-legacy"]["S\|report\|accuracy"]["lo"]` |  |
| `{{FIG:q35.hi.S_accuracy}}` | `report.json` | `report.json["q35-new-minus-legacy"]["S\|report\|accuracy"]["hi"]` |  |
| `{{FIG:q35.n.I_accuracy}}` | `report.json` | `report.json["q35-new-minus-legacy"]["I\|report\|accuracy"]["n"]` |  |
| `{{FIG:q35.diff.I_accuracy}}` | `report.json` | `report.json["q35-new-minus-legacy"]["I\|report\|accuracy"]["diff"]` |  |
| `{{FIG:q35.lo.I_accuracy}}` | `report.json` | `report.json["q35-new-minus-legacy"]["I\|report\|accuracy"]["lo"]` |  |
| `{{FIG:q35.hi.I_accuracy}}` | `report.json` | `report.json["q35-new-minus-legacy"]["I\|report\|accuracy"]["hi"]` |  |
| `{{FIG:q35.n.S_logloss}}` | `report.json` | `report.json["q35-new-minus-legacy"]["S\|report\|logloss"]["n"]` |  |
| `{{FIG:q35.diff.S_logloss}}` | `report.json` | `report.json["q35-new-minus-legacy"]["S\|report\|logloss"]["diff"]` |  |
| `{{FIG:q35.lo.S_logloss}}` | `report.json` | `report.json["q35-new-minus-legacy"]["S\|report\|logloss"]["lo"]` |  |
| `{{FIG:q35.hi.S_logloss}}` | `report.json` | `report.json["q35-new-minus-legacy"]["S\|report\|logloss"]["hi"]` |  |
| `{{FIG:q35.n.I_logloss}}` | `report.json` | `report.json["q35-new-minus-legacy"]["I\|report\|logloss"]["n"]` |  |
| `{{FIG:q35.diff.I_logloss}}` | `report.json` | `report.json["q35-new-minus-legacy"]["I\|report\|logloss"]["diff"]` |  |
| `{{FIG:q35.lo.I_logloss}}` | `report.json` | `report.json["q35-new-minus-legacy"]["I\|report\|logloss"]["lo"]` |  |
| `{{FIG:q35.hi.I_logloss}}` | `report.json` | `report.json["q35-new-minus-legacy"]["I\|report\|logloss"]["hi"]` |  |
| `{{FIG:q35.T.new}}` | `results/m/q35/new/fitT.json` | `T` |  |
| `{{FIG:q35.T.legacy}}` | `results/m/q35/legacy/fitT.json` | `T` |  |
| `{{FIG:ts.n}}` | `baseline/metrics_v13.json` | `150 requests, the reporting half of the eval set (EVAL_CARD.md section 10.1)` |  |
| `{{FIG:ts.n.S.noul}}` | `baseline/metrics_v13.json` | `metrics["S.noul\|report"]["noul"]["n"]` |  |
| `{{FIG:ts.S.noul.accuracy}}` | `baseline/metrics_v13.json` | `metrics["S.noul\|report"]["noul"]["accuracy"]` |  |
| `{{FIG:tsc.b12.S.noul.accuracy}}` | `report.json` | `["baseline"]-section: model b12 on the baseline ids, S.noul.accuracy` | model scored on exactly the baseline's ids (REPORT.md section 4); budget as chosen as default |
| `{{FIG:tsc.b31.S.noul.accuracy}}` | `report.json` | `["baseline"]-section: model b31 on the baseline ids, S.noul.accuracy` | model scored on exactly the baseline's ids (REPORT.md section 4); budget as chosen as default |
| `{{FIG:ts.n.S.choice}}` | `baseline/metrics_v13.json` | `metrics["S.choice\|report"]["choice"]["n"]` |  |
| `{{FIG:ts.S.choice.accuracy}}` | `baseline/metrics_v13.json` | `metrics["S.choice\|report"]["choice"]["accuracy"]` |  |
| `{{FIG:tsc.b12.S.choice.accuracy}}` | `report.json` | `["baseline"]-section: model b12 on the baseline ids, S.choice.accuracy` | model scored on exactly the baseline's ids (REPORT.md section 4); budget as chosen as default |
| `{{FIG:tsc.b31.S.choice.accuracy}}` | `report.json` | `["baseline"]-section: model b31 on the baseline ids, S.choice.accuracy` | model scored on exactly the baseline's ids (REPORT.md section 4); budget as chosen as default |
| `{{FIG:ts.n.S.score}}` | `baseline/metrics_v13.json` | `metrics["S.score\|report"]["score"]["n"]` |  |
| `{{FIG:ts.S.score.accuracy}}` | `baseline/metrics_v13.json` | `metrics["S.score\|report"]["score"]["accuracy"]` |  |
| `{{FIG:tsc.b12.S.score.accuracy}}` | `report.json` | `["baseline"]-section: model b12 on the baseline ids, S.score.accuracy` | model scored on exactly the baseline's ids (REPORT.md section 4); budget as chosen as default |
| `{{FIG:tsc.b31.S.score.accuracy}}` | `report.json` | `["baseline"]-section: model b31 on the baseline ids, S.score.accuracy` | model scored on exactly the baseline's ids (REPORT.md section 4); budget as chosen as default |
| `{{FIG:ts.n.R.noul}}` | `baseline/metrics_v13.json` | `metrics["R.noul\|report"]["noul"]["n"]` |  |
| `{{FIG:ts.R.noul.accuracy}}` | `baseline/metrics_v13.json` | `metrics["R.noul\|report"]["noul"]["accuracy"]` |  |
| `{{FIG:tsc.b12.R.noul.accuracy}}` | `report.json` | `["baseline"]-section: model b12 on the baseline ids, R.noul.accuracy` | model scored on exactly the baseline's ids (REPORT.md section 4); budget as chosen as default |
| `{{FIG:tsc.b31.R.noul.accuracy}}` | `report.json` | `["baseline"]-section: model b31 on the baseline ids, R.noul.accuracy` | model scored on exactly the baseline's ids (REPORT.md section 4); budget as chosen as default |
| `{{FIG:ts.n.R.choice}}` | `baseline/metrics_v13.json` | `metrics["R.choice\|report"]["choice"]["n"]` |  |
| `{{FIG:ts.R.choice.accuracy}}` | `baseline/metrics_v13.json` | `metrics["R.choice\|report"]["choice"]["accuracy"]` |  |
| `{{FIG:tsc.b12.R.choice.accuracy}}` | `report.json` | `["baseline"]-section: model b12 on the baseline ids, R.choice.accuracy` | model scored on exactly the baseline's ids (REPORT.md section 4); budget as chosen as default |
| `{{FIG:tsc.b31.R.choice.accuracy}}` | `report.json` | `["baseline"]-section: model b31 on the baseline ids, R.choice.accuracy` | model scored on exactly the baseline's ids (REPORT.md section 4); budget as chosen as default |
| `{{FIG:ts.n.S.box}}` | `baseline/metrics_v13.json` | `metrics["S.box\|report"]["box"]["n"]` |  |
| `{{FIG:ts.S.box.mean_iou}}` | `baseline/metrics_v13.json` | `metrics["S.box\|report"]["box"]["mean_iou"]` |  |
| `{{FIG:tsc.b12.S.box.mean_iou}}` | `report.json` | `["baseline"]-section: model b12 on the baseline ids, S.box.mean_iou` | model scored on exactly the baseline's ids (REPORT.md section 4); budget as chosen as default |
| `{{FIG:tsc.b31.S.box.mean_iou}}` | `report.json` | `["baseline"]-section: model b31 on the baseline ids, S.box.mean_iou` | model scored on exactly the baseline's ids (REPORT.md section 4); budget as chosen as default |
| `{{FIG:ts.S.box.iou50}}` | `baseline/metrics_v13.json` | `metrics["S.box\|report"]["box"]["iou50"]` |  |
| `{{FIG:tsc.b12.S.box.iou50}}` | `report.json` | `["baseline"]-section: model b12 on the baseline ids, S.box.iou50` | model scored on exactly the baseline's ids (REPORT.md section 4); budget as chosen as default |
| `{{FIG:tsc.b31.S.box.iou50}}` | `report.json` | `["baseline"]-section: model b31 on the baseline ids, S.box.iou50` | model scored on exactly the baseline's ids (REPORT.md section 4); budget as chosen as default |
| `{{FIG:ts.S.box.conf_ece}}` | `baseline/metrics_v13.json` | `metrics["S.box\|report"]["box"]["conf_ece"]` |  |
| `{{FIG:tsc.b12.S.box.conf_ece}}` | `report.json` | `["baseline"]-section: model b12 on the baseline ids, S.box.conf_ece` | model scored on exactly the baseline's ids (REPORT.md section 4); budget as chosen as default |
| `{{FIG:tsc.b31.S.box.conf_ece}}` | `report.json` | `["baseline"]-section: model b31 on the baseline ids, S.box.conf_ece` | model scored on exactly the baseline's ids (REPORT.md section 4); budget as chosen as default |
| `{{FIG:ts.S.box.absent_auroc}}` | `baseline/metrics_v13.json` | `metrics["S.box\|report"]["box"]["absent_auroc"]` |  |
| `{{FIG:tsc.b12.S.box.absent_auroc}}` | `report.json` | `["baseline"]-section: model b12 on the baseline ids, S.box.absent_auroc` | model scored on exactly the baseline's ids (REPORT.md section 4); budget as chosen as default |
| `{{FIG:tsc.b31.S.box.absent_auroc}}` | `report.json` | `["baseline"]-section: model b31 on the baseline ids, S.box.absent_auroc` | model scored on exactly the baseline's ids (REPORT.md section 4); budget as chosen as default |
| `{{FIG:ts.n.R.box}}` | `baseline/metrics_v13.json` | `metrics["R.box\|report"]["box"]["n"]` |  |
| `{{FIG:ts.R.box.mean_iou}}` | `baseline/metrics_v13.json` | `metrics["R.box\|report"]["box"]["mean_iou"]` |  |
| `{{FIG:tsc.b12.R.box.mean_iou}}` | `report.json` | `["baseline"]-section: model b12 on the baseline ids, R.box.mean_iou` | model scored on exactly the baseline's ids (REPORT.md section 4); budget as chosen as default |
| `{{FIG:tsc.b31.R.box.mean_iou}}` | `report.json` | `["baseline"]-section: model b31 on the baseline ids, R.box.mean_iou` | model scored on exactly the baseline's ids (REPORT.md section 4); budget as chosen as default |
| `{{FIG:ts.R.box.iou50}}` | `baseline/metrics_v13.json` | `metrics["R.box\|report"]["box"]["iou50"]` |  |
| `{{FIG:tsc.b12.R.box.iou50}}` | `report.json` | `["baseline"]-section: model b12 on the baseline ids, R.box.iou50` | model scored on exactly the baseline's ids (REPORT.md section 4); budget as chosen as default |
| `{{FIG:tsc.b31.R.box.iou50}}` | `report.json` | `["baseline"]-section: model b31 on the baseline ids, R.box.iou50` | model scored on exactly the baseline's ids (REPORT.md section 4); budget as chosen as default |

Total distinct slots: 435.

## Gaps in the session's report (no report key)

1. `box_companion_auroc.*`: `report.py` scores the Box answer's `confidence` only. The companion noul's AUROC over present and absent S Box items needs the run to ask a paired noul per Box item (or the filler to compute it from the S noul answers if the eval set carries one). No key exists.
2. `m.<m>.<b>.S.qimg.*`: `report.py` prints S.qimg per answer kind (noul and choice), not pooled; the card wants one pooled figure. Add a pooled row (from `score.per_item_vectors`) or state both kinds.
3. `parity.<m>.verdict`: computed in the report's text, not stored in `report.json`. The filler must re-derive it by the stated rule.
4. `pair.gate_verdict`: also text only (REPORT.md section 2).
5. `tree.*`: the check's output (`l6.py tree --out F`) is not read by `report.py`; the filler reads the file directly (its path in the run is a guess: `results/tree_b12.json`).
6. `ts.*` and `tsc.*`: `report.py` section 4 prints the model against the baseline per budget as markdown; `report.json` keeps only `["baseline"]` metrics, not the model's on the baseline ids. `tsc.*` needs those per-model metrics saved to json.
7. `run.power_limit_w`, `run.decisio_sha`, `run.folder`, `run.date`, `run.m16_budget`: from the run's setup files and folder name, not from `report.py`.
8. The Qwen stage's `fitT.json` (`q35.T.*`) is written by the run, not read by `report.py` (it prints the first 300 characters).
9. `lat.*`: the numbers come from `latency_table.py`, which prints markdown; the filler reads `latency.json` directly or parses that table. `lat.*.box` needs the Box stage to have run (PR C head).

## Code values that change once the figures exist (TODO; no code changed here)

- TODO: the default image budget (`--image-budget` default; `decision.default_budget`): choose from 10.2, 10.3 and 10.5; the server's default, `docs/api.md` (budget table and the `--images on` paragraph), `docs/running.md`, `docs/cli.md`/flags reference.
- TODO: refuse `--image-budget 1120` on a Gemma base at start if the corner probe fails on the 31B or the tuned 12B as well (10.5 says it is refused; the code must match), with the message naming the reason.
- TODO: each family's `image_temperature` in `families.py` (12B base, tuned 12B and its adapters, 31B base): set from `fitT.<m>.<b>.T` for the default budget. `docs/api.md` still says none is fitted.
- TODO: ship the Box confidence map per model (`box_confidence.json` in the model's folder or the repository's defaults) from the calibration half (`box.confidence_map`). Until then `x-decisio-box` says `calibrated=false`, and `docs/box.md` and `docs/api.md` must keep that wording.
- TODO: the Qwen route's default layout and the deprecation notice if the new layout is not better (10.10).

## Public text to revise when the figures land

- `docs/api.md`: the lines saying memory, text latency and read quality "are not measured yet", and that no `image_temperature` is fitted; the line saying the image route is measured with the Qwen base only; the `Images:` paragraph near the top.
- `docs/box.md`, last bullet: "How accurate the boxes are ... are not measured yet".
- `EVAL_CARD.md` 7.1 and 7.5: "the image route was not measured" and "not served on a Mac"; 6.x for the 12B likewise.
- `README.md`: the capability table row for images, and the ImajevBench row (0.791, Qwen route) if the Gemma figures replace it.

## Open questions

- Is the eval set itself published (items, scorer, checksums)? 10.1 says every file is in the run's record and its sha256 list, and does not offer the items; if they should be published, say where.
- The TypeSafe row "Box on a request with no image": TypeSafe's documentation says a Box needs an image; its behaviour without one was not probed. Probe once before merging, or keep the wording.
- `docs/comparison.md` is dated 2026-10-05 throughout; section 3.1 is dated 2026-11-14. The snapshot file `snapshots/typesafe_docs.json` has no image pages of 2026-11-14; add them to the run if the docs are to quote them.
- Which budget the TypeSafe comparison table uses for decisio's columns (`decision.default_budget`) until the default is decided.
- Section 10.4 reports the tuned 12B only; add a 31B minus 12B table if wanted.
