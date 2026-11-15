NOT FOR MERGING: maps every `{{FIG:...}}` slot of the three docs to where its value comes from; `fill_records.py` reads it. Delete this file before the records PR is opened.

Sources: `report` is a path in the run's `report.json` (keys joined by ` > `; the report's own keys such as `b12/280` and `S.noul\|report` are single keys); `file` is a json file in the run folder; `calc` is a short computation over the run folder's files; `manual` is a decision given with `--set`.
`{m16}` is `run.m16_budget` of the report (the tuned 12B's single budget).
Formats: `r3` three decimals; `r4` four decimals; `sd3` signed, three decimals; `n` whole number with thousands separators; `pair` `p50 / p95` in whole milliseconds; `w` watts without decimals; `folder` the last part of the path.
A `calc` or `file` source needs `--results` (the run folder).

| Slot | Source | Path | Format | Note |
| --- | --- | --- | --- | --- |
| `{{FIG:run.power_limit_w}}` | file | `first_contact.json > power_limit_w` | w |  |
| `{{FIG:run.decisio_sha}}` | calc | `decisio_sha` | text | last checkout the run recorded |
| `{{FIG:run.m16_budget}}` | report | `run > m16_budget` | text |  |
| `{{FIG:run.date}}` | calc | `run_date` | text | US Pacific date of the first stage, else of the report |
| `{{FIG:run.folder}}` | report | `run > run_folder` | folder |  |
| `{{FIG:n.S.noul}}` | report | `b12/280 > metrics > S.noul\|report > noul > n` | n | the same items for every model |
| `{{FIG:m.b12.280.S.noul.accuracy}}` | report | `b12/280 > metrics > S.noul\|report > noul > accuracy` | r3 |  |
| `{{FIG:m.b12.280.S.noul.logloss}}` | report | `b12/280 > metrics > S.noul\|report > noul > logloss` | r3 |  |
| `{{FIG:m.b12.280.S.noul.ece}}` | report | `b12/280 > metrics > S.noul\|report > noul > ece` | r3 |  |
| `{{FIG:m.b12.1120.S.noul.accuracy}}` | report | `b12/1120 > metrics > S.noul\|report > noul > accuracy` | r3 |  |
| `{{FIG:m.b12.1120.S.noul.logloss}}` | report | `b12/1120 > metrics > S.noul\|report > noul > logloss` | r3 |  |
| `{{FIG:m.b12.1120.S.noul.ece}}` | report | `b12/1120 > metrics > S.noul\|report > noul > ece` | r3 |  |
| `{{FIG:m.m16.sel.S.noul.accuracy}}` | report | `m16/{m16} > metrics > S.noul\|report > noul > accuracy` | r3 |  |
| `{{FIG:m.m16.sel.S.noul.logloss}}` | report | `m16/{m16} > metrics > S.noul\|report > noul > logloss` | r3 |  |
| `{{FIG:m.m16.sel.S.noul.ece}}` | report | `m16/{m16} > metrics > S.noul\|report > noul > ece` | r3 |  |
| `{{FIG:m.b31.280.S.noul.accuracy}}` | report | `b31/280 > metrics > S.noul\|report > noul > accuracy` | r3 |  |
| `{{FIG:m.b31.280.S.noul.logloss}}` | report | `b31/280 > metrics > S.noul\|report > noul > logloss` | r3 |  |
| `{{FIG:m.b31.280.S.noul.ece}}` | report | `b31/280 > metrics > S.noul\|report > noul > ece` | r3 |  |
| `{{FIG:m.b31.1120.S.noul.accuracy}}` | report | `b31/1120 > metrics > S.noul\|report > noul > accuracy` | r3 |  |
| `{{FIG:m.b31.1120.S.noul.logloss}}` | report | `b31/1120 > metrics > S.noul\|report > noul > logloss` | r3 |  |
| `{{FIG:m.b31.1120.S.noul.ece}}` | report | `b31/1120 > metrics > S.noul\|report > noul > ece` | r3 |  |
| `{{FIG:n.S.choice}}` | report | `b12/280 > metrics > S.choice\|report > choice > n` | n | the same items for every model |
| `{{FIG:m.b12.280.S.choice.accuracy}}` | report | `b12/280 > metrics > S.choice\|report > choice > accuracy` | r3 |  |
| `{{FIG:m.b12.280.S.choice.logloss}}` | report | `b12/280 > metrics > S.choice\|report > choice > logloss` | r3 |  |
| `{{FIG:m.b12.280.S.choice.ece}}` | report | `b12/280 > metrics > S.choice\|report > choice > ece` | r3 |  |
| `{{FIG:m.b12.1120.S.choice.accuracy}}` | report | `b12/1120 > metrics > S.choice\|report > choice > accuracy` | r3 |  |
| `{{FIG:m.b12.1120.S.choice.logloss}}` | report | `b12/1120 > metrics > S.choice\|report > choice > logloss` | r3 |  |
| `{{FIG:m.b12.1120.S.choice.ece}}` | report | `b12/1120 > metrics > S.choice\|report > choice > ece` | r3 |  |
| `{{FIG:m.m16.sel.S.choice.accuracy}}` | report | `m16/{m16} > metrics > S.choice\|report > choice > accuracy` | r3 |  |
| `{{FIG:m.m16.sel.S.choice.logloss}}` | report | `m16/{m16} > metrics > S.choice\|report > choice > logloss` | r3 |  |
| `{{FIG:m.m16.sel.S.choice.ece}}` | report | `m16/{m16} > metrics > S.choice\|report > choice > ece` | r3 |  |
| `{{FIG:m.b31.280.S.choice.accuracy}}` | report | `b31/280 > metrics > S.choice\|report > choice > accuracy` | r3 |  |
| `{{FIG:m.b31.280.S.choice.logloss}}` | report | `b31/280 > metrics > S.choice\|report > choice > logloss` | r3 |  |
| `{{FIG:m.b31.280.S.choice.ece}}` | report | `b31/280 > metrics > S.choice\|report > choice > ece` | r3 |  |
| `{{FIG:m.b31.1120.S.choice.accuracy}}` | report | `b31/1120 > metrics > S.choice\|report > choice > accuracy` | r3 |  |
| `{{FIG:m.b31.1120.S.choice.logloss}}` | report | `b31/1120 > metrics > S.choice\|report > choice > logloss` | r3 |  |
| `{{FIG:m.b31.1120.S.choice.ece}}` | report | `b31/1120 > metrics > S.choice\|report > choice > ece` | r3 |  |
| `{{FIG:n.S.score}}` | report | `b12/280 > metrics > S.score\|report > score > n` | n | the same items for every model |
| `{{FIG:m.b12.280.S.score.accuracy}}` | report | `b12/280 > metrics > S.score\|report > score > accuracy` | r3 |  |
| `{{FIG:m.b12.280.S.score.logloss}}` | report | `b12/280 > metrics > S.score\|report > score > logloss` | r3 |  |
| `{{FIG:m.b12.280.S.score.ece}}` | report | `b12/280 > metrics > S.score\|report > score > ece` | r3 |  |
| `{{FIG:m.b12.1120.S.score.accuracy}}` | report | `b12/1120 > metrics > S.score\|report > score > accuracy` | r3 |  |
| `{{FIG:m.b12.1120.S.score.logloss}}` | report | `b12/1120 > metrics > S.score\|report > score > logloss` | r3 |  |
| `{{FIG:m.b12.1120.S.score.ece}}` | report | `b12/1120 > metrics > S.score\|report > score > ece` | r3 |  |
| `{{FIG:m.m16.sel.S.score.accuracy}}` | report | `m16/{m16} > metrics > S.score\|report > score > accuracy` | r3 |  |
| `{{FIG:m.m16.sel.S.score.logloss}}` | report | `m16/{m16} > metrics > S.score\|report > score > logloss` | r3 |  |
| `{{FIG:m.m16.sel.S.score.ece}}` | report | `m16/{m16} > metrics > S.score\|report > score > ece` | r3 |  |
| `{{FIG:m.b31.280.S.score.accuracy}}` | report | `b31/280 > metrics > S.score\|report > score > accuracy` | r3 |  |
| `{{FIG:m.b31.280.S.score.logloss}}` | report | `b31/280 > metrics > S.score\|report > score > logloss` | r3 |  |
| `{{FIG:m.b31.280.S.score.ece}}` | report | `b31/280 > metrics > S.score\|report > score > ece` | r3 |  |
| `{{FIG:m.b31.1120.S.score.accuracy}}` | report | `b31/1120 > metrics > S.score\|report > score > accuracy` | r3 |  |
| `{{FIG:m.b31.1120.S.score.logloss}}` | report | `b31/1120 > metrics > S.score\|report > score > logloss` | r3 |  |
| `{{FIG:m.b31.1120.S.score.ece}}` | report | `b31/1120 > metrics > S.score\|report > score > ece` | r3 |  |
| `{{FIG:n.S.qimg}}` | report | `qimg_pooled/b12/280/report > n` | n |  |
| `{{FIG:m.b12.280.S.qimg.accuracy}}` | report | `qimg_pooled/b12/280/report > accuracy` | r3 | pooled over noul, choice and score |
| `{{FIG:m.b12.280.S.qimg.logloss}}` | report | `qimg_pooled/b12/280/report > logloss` | r3 | pooled over noul, choice and score |
| `{{FIG:m.b12.1120.S.qimg.accuracy}}` | report | `qimg_pooled/b12/1120/report > accuracy` | r3 | pooled over noul, choice and score |
| `{{FIG:m.b12.1120.S.qimg.logloss}}` | report | `qimg_pooled/b12/1120/report > logloss` | r3 | pooled over noul, choice and score |
| `{{FIG:m.m16.sel.S.qimg.accuracy}}` | report | `qimg_pooled/m16/{m16}/report > accuracy` | r3 | pooled over noul, choice and score |
| `{{FIG:m.m16.sel.S.qimg.logloss}}` | report | `qimg_pooled/m16/{m16}/report > logloss` | r3 | pooled over noul, choice and score |
| `{{FIG:m.b31.280.S.qimg.accuracy}}` | report | `qimg_pooled/b31/280/report > accuracy` | r3 | pooled over noul, choice and score |
| `{{FIG:m.b31.280.S.qimg.logloss}}` | report | `qimg_pooled/b31/280/report > logloss` | r3 | pooled over noul, choice and score |
| `{{FIG:m.b31.1120.S.qimg.accuracy}}` | report | `qimg_pooled/b31/1120/report > accuracy` | r3 | pooled over noul, choice and score |
| `{{FIG:m.b31.1120.S.qimg.logloss}}` | report | `qimg_pooled/b31/1120/report > logloss` | r3 | pooled over noul, choice and score |
| `{{FIG:n.R.noul}}` | report | `b12/280 > metrics > R.noul\|report > noul > n` | n | the same items for every model |
| `{{FIG:m.b12.280.R.noul.accuracy}}` | report | `b12/280 > metrics > R.noul\|report > noul > accuracy` | r3 |  |
| `{{FIG:m.b12.280.R.noul.logloss}}` | report | `b12/280 > metrics > R.noul\|report > noul > logloss` | r3 |  |
| `{{FIG:m.b12.280.R.noul.ece}}` | report | `b12/280 > metrics > R.noul\|report > noul > ece` | r3 |  |
| `{{FIG:m.b12.1120.R.noul.accuracy}}` | report | `b12/1120 > metrics > R.noul\|report > noul > accuracy` | r3 |  |
| `{{FIG:m.b12.1120.R.noul.logloss}}` | report | `b12/1120 > metrics > R.noul\|report > noul > logloss` | r3 |  |
| `{{FIG:m.b12.1120.R.noul.ece}}` | report | `b12/1120 > metrics > R.noul\|report > noul > ece` | r3 |  |
| `{{FIG:m.m16.sel.R.noul.accuracy}}` | report | `m16/{m16} > metrics > R.noul\|report > noul > accuracy` | r3 |  |
| `{{FIG:m.m16.sel.R.noul.logloss}}` | report | `m16/{m16} > metrics > R.noul\|report > noul > logloss` | r3 |  |
| `{{FIG:m.m16.sel.R.noul.ece}}` | report | `m16/{m16} > metrics > R.noul\|report > noul > ece` | r3 |  |
| `{{FIG:m.b31.280.R.noul.accuracy}}` | report | `b31/280 > metrics > R.noul\|report > noul > accuracy` | r3 |  |
| `{{FIG:m.b31.280.R.noul.logloss}}` | report | `b31/280 > metrics > R.noul\|report > noul > logloss` | r3 |  |
| `{{FIG:m.b31.280.R.noul.ece}}` | report | `b31/280 > metrics > R.noul\|report > noul > ece` | r3 |  |
| `{{FIG:m.b31.1120.R.noul.accuracy}}` | report | `b31/1120 > metrics > R.noul\|report > noul > accuracy` | r3 |  |
| `{{FIG:m.b31.1120.R.noul.logloss}}` | report | `b31/1120 > metrics > R.noul\|report > noul > logloss` | r3 |  |
| `{{FIG:m.b31.1120.R.noul.ece}}` | report | `b31/1120 > metrics > R.noul\|report > noul > ece` | r3 |  |
| `{{FIG:n.R.choice}}` | report | `b12/280 > metrics > R.choice\|report > choice > n` | n | the same items for every model |
| `{{FIG:m.b12.280.R.choice.accuracy}}` | report | `b12/280 > metrics > R.choice\|report > choice > accuracy` | r3 |  |
| `{{FIG:m.b12.280.R.choice.logloss}}` | report | `b12/280 > metrics > R.choice\|report > choice > logloss` | r3 |  |
| `{{FIG:m.b12.280.R.choice.ece}}` | report | `b12/280 > metrics > R.choice\|report > choice > ece` | r3 |  |
| `{{FIG:m.b12.1120.R.choice.accuracy}}` | report | `b12/1120 > metrics > R.choice\|report > choice > accuracy` | r3 |  |
| `{{FIG:m.b12.1120.R.choice.logloss}}` | report | `b12/1120 > metrics > R.choice\|report > choice > logloss` | r3 |  |
| `{{FIG:m.b12.1120.R.choice.ece}}` | report | `b12/1120 > metrics > R.choice\|report > choice > ece` | r3 |  |
| `{{FIG:m.m16.sel.R.choice.accuracy}}` | report | `m16/{m16} > metrics > R.choice\|report > choice > accuracy` | r3 |  |
| `{{FIG:m.m16.sel.R.choice.logloss}}` | report | `m16/{m16} > metrics > R.choice\|report > choice > logloss` | r3 |  |
| `{{FIG:m.m16.sel.R.choice.ece}}` | report | `m16/{m16} > metrics > R.choice\|report > choice > ece` | r3 |  |
| `{{FIG:m.b31.280.R.choice.accuracy}}` | report | `b31/280 > metrics > R.choice\|report > choice > accuracy` | r3 |  |
| `{{FIG:m.b31.280.R.choice.logloss}}` | report | `b31/280 > metrics > R.choice\|report > choice > logloss` | r3 |  |
| `{{FIG:m.b31.280.R.choice.ece}}` | report | `b31/280 > metrics > R.choice\|report > choice > ece` | r3 |  |
| `{{FIG:m.b31.1120.R.choice.accuracy}}` | report | `b31/1120 > metrics > R.choice\|report > choice > accuracy` | r3 |  |
| `{{FIG:m.b31.1120.R.choice.logloss}}` | report | `b31/1120 > metrics > R.choice\|report > choice > logloss` | r3 |  |
| `{{FIG:m.b31.1120.R.choice.ece}}` | report | `b31/1120 > metrics > R.choice\|report > choice > ece` | r3 |  |
| `{{FIG:n.I.noul}}` | report | `b12/280 > metrics > I.noul\|report > noul > n` | n | the same items for every model |
| `{{FIG:m.b12.280.I.noul.accuracy}}` | report | `b12/280 > metrics > I.noul\|report > noul > accuracy` | r3 |  |
| `{{FIG:m.b12.280.I.noul.logloss}}` | report | `b12/280 > metrics > I.noul\|report > noul > logloss` | r3 |  |
| `{{FIG:m.b12.280.I.noul.ece}}` | report | `b12/280 > metrics > I.noul\|report > noul > ece` | r3 |  |
| `{{FIG:m.b12.1120.I.noul.accuracy}}` | report | `b12/1120 > metrics > I.noul\|report > noul > accuracy` | r3 |  |
| `{{FIG:m.b12.1120.I.noul.logloss}}` | report | `b12/1120 > metrics > I.noul\|report > noul > logloss` | r3 |  |
| `{{FIG:m.b12.1120.I.noul.ece}}` | report | `b12/1120 > metrics > I.noul\|report > noul > ece` | r3 |  |
| `{{FIG:m.m16.sel.I.noul.accuracy}}` | report | `m16/{m16} > metrics > I.noul\|report > noul > accuracy` | r3 |  |
| `{{FIG:m.m16.sel.I.noul.logloss}}` | report | `m16/{m16} > metrics > I.noul\|report > noul > logloss` | r3 |  |
| `{{FIG:m.m16.sel.I.noul.ece}}` | report | `m16/{m16} > metrics > I.noul\|report > noul > ece` | r3 |  |
| `{{FIG:m.b31.280.I.noul.accuracy}}` | report | `b31/280 > metrics > I.noul\|report > noul > accuracy` | r3 |  |
| `{{FIG:m.b31.280.I.noul.logloss}}` | report | `b31/280 > metrics > I.noul\|report > noul > logloss` | r3 |  |
| `{{FIG:m.b31.280.I.noul.ece}}` | report | `b31/280 > metrics > I.noul\|report > noul > ece` | r3 |  |
| `{{FIG:m.b31.1120.I.noul.accuracy}}` | report | `b31/1120 > metrics > I.noul\|report > noul > accuracy` | r3 |  |
| `{{FIG:m.b31.1120.I.noul.logloss}}` | report | `b31/1120 > metrics > I.noul\|report > noul > logloss` | r3 |  |
| `{{FIG:m.b31.1120.I.noul.ece}}` | report | `b31/1120 > metrics > I.noul\|report > noul > ece` | r3 |  |
| `{{FIG:n.I.choice}}` | report | `b12/280 > metrics > I.choice\|report > choice > n` | n | the same items for every model |
| `{{FIG:m.b12.280.I.choice.accuracy}}` | report | `b12/280 > metrics > I.choice\|report > choice > accuracy` | r3 |  |
| `{{FIG:m.b12.280.I.choice.logloss}}` | report | `b12/280 > metrics > I.choice\|report > choice > logloss` | r3 |  |
| `{{FIG:m.b12.280.I.choice.ece}}` | report | `b12/280 > metrics > I.choice\|report > choice > ece` | r3 |  |
| `{{FIG:m.b12.1120.I.choice.accuracy}}` | report | `b12/1120 > metrics > I.choice\|report > choice > accuracy` | r3 |  |
| `{{FIG:m.b12.1120.I.choice.logloss}}` | report | `b12/1120 > metrics > I.choice\|report > choice > logloss` | r3 |  |
| `{{FIG:m.b12.1120.I.choice.ece}}` | report | `b12/1120 > metrics > I.choice\|report > choice > ece` | r3 |  |
| `{{FIG:m.m16.sel.I.choice.accuracy}}` | report | `m16/{m16} > metrics > I.choice\|report > choice > accuracy` | r3 |  |
| `{{FIG:m.m16.sel.I.choice.logloss}}` | report | `m16/{m16} > metrics > I.choice\|report > choice > logloss` | r3 |  |
| `{{FIG:m.m16.sel.I.choice.ece}}` | report | `m16/{m16} > metrics > I.choice\|report > choice > ece` | r3 |  |
| `{{FIG:m.b31.280.I.choice.accuracy}}` | report | `b31/280 > metrics > I.choice\|report > choice > accuracy` | r3 |  |
| `{{FIG:m.b31.280.I.choice.logloss}}` | report | `b31/280 > metrics > I.choice\|report > choice > logloss` | r3 |  |
| `{{FIG:m.b31.280.I.choice.ece}}` | report | `b31/280 > metrics > I.choice\|report > choice > ece` | r3 |  |
| `{{FIG:m.b31.1120.I.choice.accuracy}}` | report | `b31/1120 > metrics > I.choice\|report > choice > accuracy` | r3 |  |
| `{{FIG:m.b31.1120.I.choice.logloss}}` | report | `b31/1120 > metrics > I.choice\|report > choice > logloss` | r3 |  |
| `{{FIG:m.b31.1120.I.choice.ece}}` | report | `b31/1120 > metrics > I.choice\|report > choice > ece` | r3 |  |
| `{{FIG:n.I.score}}` | report | `b12/280 > metrics > I.score\|report > score > n` | n | the same items for every model |
| `{{FIG:m.b12.280.I.score.accuracy}}` | report | `b12/280 > metrics > I.score\|report > score > accuracy` | r3 |  |
| `{{FIG:m.b12.280.I.score.logloss}}` | report | `b12/280 > metrics > I.score\|report > score > logloss` | r3 |  |
| `{{FIG:m.b12.280.I.score.ece}}` | report | `b12/280 > metrics > I.score\|report > score > ece` | r3 |  |
| `{{FIG:m.b12.1120.I.score.accuracy}}` | report | `b12/1120 > metrics > I.score\|report > score > accuracy` | r3 |  |
| `{{FIG:m.b12.1120.I.score.logloss}}` | report | `b12/1120 > metrics > I.score\|report > score > logloss` | r3 |  |
| `{{FIG:m.b12.1120.I.score.ece}}` | report | `b12/1120 > metrics > I.score\|report > score > ece` | r3 |  |
| `{{FIG:m.m16.sel.I.score.accuracy}}` | report | `m16/{m16} > metrics > I.score\|report > score > accuracy` | r3 |  |
| `{{FIG:m.m16.sel.I.score.logloss}}` | report | `m16/{m16} > metrics > I.score\|report > score > logloss` | r3 |  |
| `{{FIG:m.m16.sel.I.score.ece}}` | report | `m16/{m16} > metrics > I.score\|report > score > ece` | r3 |  |
| `{{FIG:m.b31.280.I.score.accuracy}}` | report | `b31/280 > metrics > I.score\|report > score > accuracy` | r3 |  |
| `{{FIG:m.b31.280.I.score.logloss}}` | report | `b31/280 > metrics > I.score\|report > score > logloss` | r3 |  |
| `{{FIG:m.b31.280.I.score.ece}}` | report | `b31/280 > metrics > I.score\|report > score > ece` | r3 |  |
| `{{FIG:m.b31.1120.I.score.accuracy}}` | report | `b31/1120 > metrics > I.score\|report > score > accuracy` | r3 |  |
| `{{FIG:m.b31.1120.I.score.logloss}}` | report | `b31/1120 > metrics > I.score\|report > score > logloss` | r3 |  |
| `{{FIG:m.b31.1120.I.score.ece}}` | report | `b31/1120 > metrics > I.score\|report > score > ece` | r3 |  |
| `{{FIG:m.b12.280.S.score.mae}}` | report | `b12/280 > metrics > S.score\|report > score > mae` | r3 |  |
| `{{FIG:n_present.S.box}}` | report | `b12/280 > metrics > S.box\|report > box > n_present` | n |  |
| `{{FIG:m.b12.280.S.box.mean_iou}}` | report | `b12/280 > metrics > S.box\|report > box > mean_iou` | r3 |  |
| `{{FIG:m.b12.1120.S.box.mean_iou}}` | report | `b12/1120 > metrics > S.box\|report > box > mean_iou` | r3 |  |
| `{{FIG:m.m16.sel.S.box.mean_iou}}` | report | `m16/{m16} > metrics > S.box\|report > box > mean_iou` | r3 |  |
| `{{FIG:m.b31.280.S.box.mean_iou}}` | report | `b31/280 > metrics > S.box\|report > box > mean_iou` | r3 |  |
| `{{FIG:m.b31.1120.S.box.mean_iou}}` | report | `b31/1120 > metrics > S.box\|report > box > mean_iou` | r3 |  |
| `{{FIG:m.b12.280.S.box.iou50}}` | report | `b12/280 > metrics > S.box\|report > box > iou50` | r3 |  |
| `{{FIG:m.b12.1120.S.box.iou50}}` | report | `b12/1120 > metrics > S.box\|report > box > iou50` | r3 |  |
| `{{FIG:m.m16.sel.S.box.iou50}}` | report | `m16/{m16} > metrics > S.box\|report > box > iou50` | r3 |  |
| `{{FIG:m.b31.280.S.box.iou50}}` | report | `b31/280 > metrics > S.box\|report > box > iou50` | r3 |  |
| `{{FIG:m.b31.1120.S.box.iou50}}` | report | `b31/1120 > metrics > S.box\|report > box > iou50` | r3 |  |
| `{{FIG:m.b12.280.S.box.iou75}}` | report | `b12/280 > metrics > S.box\|report > box > iou75` | r3 |  |
| `{{FIG:m.b12.1120.S.box.iou75}}` | report | `b12/1120 > metrics > S.box\|report > box > iou75` | r3 |  |
| `{{FIG:m.m16.sel.S.box.iou75}}` | report | `m16/{m16} > metrics > S.box\|report > box > iou75` | r3 |  |
| `{{FIG:m.b31.280.S.box.iou75}}` | report | `b31/280 > metrics > S.box\|report > box > iou75` | r3 |  |
| `{{FIG:m.b31.1120.S.box.iou75}}` | report | `b31/1120 > metrics > S.box\|report > box > iou75` | r3 |  |
| `{{FIG:m.b12.280.S.box.edge_mae}}` | report | `b12/280 > metrics > S.box\|report > box > edge_mae` | r3 |  |
| `{{FIG:m.b12.1120.S.box.edge_mae}}` | report | `b12/1120 > metrics > S.box\|report > box > edge_mae` | r3 |  |
| `{{FIG:m.m16.sel.S.box.edge_mae}}` | report | `m16/{m16} > metrics > S.box\|report > box > edge_mae` | r3 |  |
| `{{FIG:m.b31.280.S.box.edge_mae}}` | report | `b31/280 > metrics > S.box\|report > box > edge_mae` | r3 |  |
| `{{FIG:m.b31.1120.S.box.edge_mae}}` | report | `b31/1120 > metrics > S.box\|report > box > edge_mae` | r3 |  |
| `{{FIG:m.b12.280.S.box.conf_ece}}` | report | `b12/280 > metrics > S.box\|report > box > conf_ece` | r3 |  |
| `{{FIG:m.b12.1120.S.box.conf_ece}}` | report | `b12/1120 > metrics > S.box\|report > box > conf_ece` | r3 |  |
| `{{FIG:m.m16.sel.S.box.conf_ece}}` | report | `m16/{m16} > metrics > S.box\|report > box > conf_ece` | r3 |  |
| `{{FIG:m.b31.280.S.box.conf_ece}}` | report | `b31/280 > metrics > S.box\|report > box > conf_ece` | r3 |  |
| `{{FIG:m.b31.1120.S.box.conf_ece}}` | report | `b31/1120 > metrics > S.box\|report > box > conf_ece` | r3 |  |
| `{{FIG:n_present.R.box}}` | report | `b12/280 > metrics > R.box\|report > box > n_present` | n |  |
| `{{FIG:m.b12.280.R.box.mean_iou}}` | report | `b12/280 > metrics > R.box\|report > box > mean_iou` | r3 |  |
| `{{FIG:m.b12.1120.R.box.mean_iou}}` | report | `b12/1120 > metrics > R.box\|report > box > mean_iou` | r3 |  |
| `{{FIG:m.m16.sel.R.box.mean_iou}}` | report | `m16/{m16} > metrics > R.box\|report > box > mean_iou` | r3 |  |
| `{{FIG:m.b31.280.R.box.mean_iou}}` | report | `b31/280 > metrics > R.box\|report > box > mean_iou` | r3 |  |
| `{{FIG:m.b31.1120.R.box.mean_iou}}` | report | `b31/1120 > metrics > R.box\|report > box > mean_iou` | r3 |  |
| `{{FIG:m.b12.280.R.box.iou50}}` | report | `b12/280 > metrics > R.box\|report > box > iou50` | r3 |  |
| `{{FIG:m.b12.1120.R.box.iou50}}` | report | `b12/1120 > metrics > R.box\|report > box > iou50` | r3 |  |
| `{{FIG:m.m16.sel.R.box.iou50}}` | report | `m16/{m16} > metrics > R.box\|report > box > iou50` | r3 |  |
| `{{FIG:m.b31.280.R.box.iou50}}` | report | `b31/280 > metrics > R.box\|report > box > iou50` | r3 |  |
| `{{FIG:m.b31.1120.R.box.iou50}}` | report | `b31/1120 > metrics > R.box\|report > box > iou50` | r3 |  |
| `{{FIG:m.b12.280.R.box.iou75}}` | report | `b12/280 > metrics > R.box\|report > box > iou75` | r3 |  |
| `{{FIG:m.b12.1120.R.box.iou75}}` | report | `b12/1120 > metrics > R.box\|report > box > iou75` | r3 |  |
| `{{FIG:m.m16.sel.R.box.iou75}}` | report | `m16/{m16} > metrics > R.box\|report > box > iou75` | r3 |  |
| `{{FIG:m.b31.280.R.box.iou75}}` | report | `b31/280 > metrics > R.box\|report > box > iou75` | r3 |  |
| `{{FIG:m.b31.1120.R.box.iou75}}` | report | `b31/1120 > metrics > R.box\|report > box > iou75` | r3 |  |
| `{{FIG:m.b12.280.R.box.edge_mae}}` | report | `b12/280 > metrics > R.box\|report > box > edge_mae` | r3 |  |
| `{{FIG:m.b12.1120.R.box.edge_mae}}` | report | `b12/1120 > metrics > R.box\|report > box > edge_mae` | r3 |  |
| `{{FIG:m.m16.sel.R.box.edge_mae}}` | report | `m16/{m16} > metrics > R.box\|report > box > edge_mae` | r3 |  |
| `{{FIG:m.b31.280.R.box.edge_mae}}` | report | `b31/280 > metrics > R.box\|report > box > edge_mae` | r3 |  |
| `{{FIG:m.b31.1120.R.box.edge_mae}}` | report | `b31/1120 > metrics > R.box\|report > box > edge_mae` | r3 |  |
| `{{FIG:m.b12.280.R.box.conf_ece}}` | report | `b12/280 > metrics > R.box\|report > box > conf_ece` | r3 |  |
| `{{FIG:m.b12.1120.R.box.conf_ece}}` | report | `b12/1120 > metrics > R.box\|report > box > conf_ece` | r3 |  |
| `{{FIG:m.m16.sel.R.box.conf_ece}}` | report | `m16/{m16} > metrics > R.box\|report > box > conf_ece` | r3 |  |
| `{{FIG:m.b31.280.R.box.conf_ece}}` | report | `b31/280 > metrics > R.box\|report > box > conf_ece` | r3 |  |
| `{{FIG:m.b31.1120.R.box.conf_ece}}` | report | `b31/1120 > metrics > R.box\|report > box > conf_ece` | r3 |  |
| `{{FIG:n_absent.S.box}}` | report | `b12/280 > metrics > S.box\|report > box > n_absent` | n |  |
| `{{FIG:m.b12.280.S.box.absent_auroc}}` | report | `b12/280 > metrics > S.box\|report > box > absent_auroc` | r3 |  |
| `{{FIG:m.b12.1120.S.box.absent_auroc}}` | report | `b12/1120 > metrics > S.box\|report > box > absent_auroc` | r3 |  |
| `{{FIG:m.m16.sel.S.box.absent_auroc}}` | report | `m16/{m16} > metrics > S.box\|report > box > absent_auroc` | r3 |  |
| `{{FIG:m.b31.280.S.box.absent_auroc}}` | report | `b31/280 > metrics > S.box\|report > box > absent_auroc` | r3 |  |
| `{{FIG:m.b31.1120.S.box.absent_auroc}}` | report | `b31/1120 > metrics > S.box\|report > box > absent_auroc` | r3 |  |
| `{{FIG:box_companion_auroc.b12.280}}` | report | `companion_auroc/b12/280 > auroc` | r3 |  |
| `{{FIG:box_companion_auroc.b12.1120}}` | report | `companion_auroc/b12/1120 > auroc` | r3 |  |
| `{{FIG:box_companion_auroc.m16.sel}}` | report | `companion_auroc/m16/{m16} > auroc` | r3 |  |
| `{{FIG:box_companion_auroc.b31.280}}` | report | `companion_auroc/b31/280 > auroc` | r3 |  |
| `{{FIG:box_companion_auroc.b31.1120}}` | report | `companion_auroc/b31/1120 > auroc` | r3 |  |
| `{{FIG:pair.n.S_accuracy}}` | report | `m16-b12/{m16} > S\|report\|accuracy > n` | n |  |
| `{{FIG:pair.diff.S_accuracy}}` | report | `m16-b12/{m16} > S\|report\|accuracy > diff` | sd3 |  |
| `{{FIG:pair.lo.S_accuracy}}` | report | `m16-b12/{m16} > S\|report\|accuracy > lo` | sd3 |  |
| `{{FIG:pair.hi.S_accuracy}}` | report | `m16-b12/{m16} > S\|report\|accuracy > hi` | sd3 |  |
| `{{FIG:pair.n.R_accuracy}}` | report | `m16-b12/{m16} > R\|report\|accuracy > n` | n |  |
| `{{FIG:pair.diff.R_accuracy}}` | report | `m16-b12/{m16} > R\|report\|accuracy > diff` | sd3 |  |
| `{{FIG:pair.lo.R_accuracy}}` | report | `m16-b12/{m16} > R\|report\|accuracy > lo` | sd3 |  |
| `{{FIG:pair.hi.R_accuracy}}` | report | `m16-b12/{m16} > R\|report\|accuracy > hi` | sd3 |  |
| `{{FIG:pair.n.I_accuracy}}` | report | `m16-b12/{m16} > I\|report\|accuracy > n` | n |  |
| `{{FIG:pair.diff.I_accuracy}}` | report | `m16-b12/{m16} > I\|report\|accuracy > diff` | sd3 |  |
| `{{FIG:pair.lo.I_accuracy}}` | report | `m16-b12/{m16} > I\|report\|accuracy > lo` | sd3 |  |
| `{{FIG:pair.hi.I_accuracy}}` | report | `m16-b12/{m16} > I\|report\|accuracy > hi` | sd3 |  |
| `{{FIG:pair.n.S_logloss}}` | report | `m16-b12/{m16} > S\|report\|logloss > n` | n |  |
| `{{FIG:pair.diff.S_logloss}}` | report | `m16-b12/{m16} > S\|report\|logloss > diff` | sd3 |  |
| `{{FIG:pair.lo.S_logloss}}` | report | `m16-b12/{m16} > S\|report\|logloss > lo` | sd3 |  |
| `{{FIG:pair.hi.S_logloss}}` | report | `m16-b12/{m16} > S\|report\|logloss > hi` | sd3 |  |
| `{{FIG:pair.n.R_logloss}}` | report | `m16-b12/{m16} > R\|report\|logloss > n` | n |  |
| `{{FIG:pair.diff.R_logloss}}` | report | `m16-b12/{m16} > R\|report\|logloss > diff` | sd3 |  |
| `{{FIG:pair.lo.R_logloss}}` | report | `m16-b12/{m16} > R\|report\|logloss > lo` | sd3 |  |
| `{{FIG:pair.hi.R_logloss}}` | report | `m16-b12/{m16} > R\|report\|logloss > hi` | sd3 |  |
| `{{FIG:pair.n.I_logloss}}` | report | `m16-b12/{m16} > I\|report\|logloss > n` | n |  |
| `{{FIG:pair.diff.I_logloss}}` | report | `m16-b12/{m16} > I\|report\|logloss > diff` | sd3 |  |
| `{{FIG:pair.lo.I_logloss}}` | report | `m16-b12/{m16} > I\|report\|logloss > lo` | sd3 |  |
| `{{FIG:pair.hi.I_logloss}}` | report | `m16-b12/{m16} > I\|report\|logloss > hi` | sd3 |  |
| `{{FIG:pair.n.S_mean_iou}}` | report | `m16-b12/{m16} > S\|report\|mean_iou > n` | n |  |
| `{{FIG:pair.diff.S_mean_iou}}` | report | `m16-b12/{m16} > S\|report\|mean_iou > diff` | sd3 |  |
| `{{FIG:pair.lo.S_mean_iou}}` | report | `m16-b12/{m16} > S\|report\|mean_iou > lo` | sd3 |  |
| `{{FIG:pair.hi.S_mean_iou}}` | report | `m16-b12/{m16} > S\|report\|mean_iou > hi` | sd3 |  |
| `{{FIG:pair.n.R_mean_iou}}` | report | `m16-b12/{m16} > R\|report\|mean_iou > n` | n |  |
| `{{FIG:pair.diff.R_mean_iou}}` | report | `m16-b12/{m16} > R\|report\|mean_iou > diff` | sd3 |  |
| `{{FIG:pair.lo.R_mean_iou}}` | report | `m16-b12/{m16} > R\|report\|mean_iou > lo` | sd3 |  |
| `{{FIG:pair.hi.R_mean_iou}}` | report | `m16-b12/{m16} > R\|report\|mean_iou > hi` | sd3 |  |
| `{{FIG:pair.n.S_iou50}}` | report | `m16-b12/{m16} > S\|report\|iou50 > n` | n |  |
| `{{FIG:pair.diff.S_iou50}}` | report | `m16-b12/{m16} > S\|report\|iou50 > diff` | sd3 |  |
| `{{FIG:pair.lo.S_iou50}}` | report | `m16-b12/{m16} > S\|report\|iou50 > lo` | sd3 |  |
| `{{FIG:pair.hi.S_iou50}}` | report | `m16-b12/{m16} > S\|report\|iou50 > hi` | sd3 |  |
| `{{FIG:pair.n.R_iou50}}` | report | `m16-b12/{m16} > R\|report\|iou50 > n` | n |  |
| `{{FIG:pair.diff.R_iou50}}` | report | `m16-b12/{m16} > R\|report\|iou50 > diff` | sd3 |  |
| `{{FIG:pair.lo.R_iou50}}` | report | `m16-b12/{m16} > R\|report\|iou50 > lo` | sd3 |  |
| `{{FIG:pair.hi.R_iou50}}` | report | `m16-b12/{m16} > R\|report\|iou50 > hi` | sd3 |  |
| `{{FIG:corner.b12.280}}` | file | `m/b12/on280/corner.json > passed` | n |  |
| `{{FIG:corner.b12.1120}}` | file | `m/b12/on1120/corner.json > passed` | n |  |
| `{{FIG:corner.m16.sel}}` | file | `m/m16/on{m16}/corner.json > passed` | n |  |
| `{{FIG:corner.b31.280}}` | file | `m/b31/on280/corner.json > passed` | n |  |
| `{{FIG:corner.b31.1120}}` | file | `m/b31/on1120/corner.json > passed` | n |  |
| `{{FIG:decision.default_budget}}` | manual | `` | text | the maintainers' decision after the run: --set decision.default_budget=N |
| `{{FIG:parity.b12.max_dp}}` | report | `parity/b12/on280 > max_dp` | r4 |  |
| `{{FIG:parity.b12.changed}}` | report | `parity/b12/on280 > choices_changed` | n |  |
| `{{FIG:parity.b12.rep_max_dp}}` | report | `parity/b12/repeat > max_dp` | r4 |  |
| `{{FIG:parity.b12.rep_changed}}` | report | `parity/b12/repeat > choices_changed` | n |  |
| `{{FIG:parity.b12.verdict}}` | report | `parity/b12/on280 > verdict` | text |  |
| `{{FIG:parity.m16.max_dp}}` | report | `parity/m16/on{m16} > max_dp` | r4 |  |
| `{{FIG:parity.m16.changed}}` | report | `parity/m16/on{m16} > choices_changed` | n |  |
| `{{FIG:parity.m16.rep_max_dp}}` | report | `parity/m16/repeat > max_dp` | r4 |  |
| `{{FIG:parity.m16.rep_changed}}` | report | `parity/m16/repeat > choices_changed` | n |  |
| `{{FIG:parity.m16.verdict}}` | report | `parity/m16/on{m16} > verdict` | text |  |
| `{{FIG:parity.b31.max_dp}}` | report | `parity/b31/on280 > max_dp` | r4 |  |
| `{{FIG:parity.b31.changed}}` | report | `parity/b31/on280 > choices_changed` | n |  |
| `{{FIG:parity.b31.rep_max_dp}}` | report | `parity/b31/repeat > max_dp` | r4 |  |
| `{{FIG:parity.b31.rep_changed}}` | report | `parity/b31/repeat > choices_changed` | n |  |
| `{{FIG:parity.b31.verdict}}` | report | `parity/b31/on280 > verdict` | text |  |
| `{{FIG:start.b12.off1.ready}}` | calc | `start b12 off1 ready` | text | the last row for that start in starts.jsonl |
| `{{FIG:start.b12.off1.seconds}}` | calc | `start b12 off1 seconds` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.b12.off1.mib}}` | calc | `start b12 off1 mib` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.b12.off1.peak}}` | calc | `start b12 off1 peak` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.b12.on280.ready}}` | calc | `start b12 on280 ready` | text | the last row for that start in starts.jsonl |
| `{{FIG:start.b12.on280.seconds}}` | calc | `start b12 on280 seconds` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.b12.on280.mib}}` | calc | `start b12 on280 mib` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.b12.on280.peak}}` | calc | `start b12 on280 peak` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.b12.on1120.ready}}` | calc | `start b12 on1120 ready` | text | the last row for that start in starts.jsonl |
| `{{FIG:start.b12.on1120.seconds}}` | calc | `start b12 on1120 seconds` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.b12.on1120.mib}}` | calc | `start b12 on1120 mib` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.b12.on1120.peak}}` | calc | `start b12 on1120 peak` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.b12.m128.ready}}` | calc | `start b12 m128 ready` | text | the last row for that start in starts.jsonl |
| `{{FIG:start.b12.m128.seconds}}` | calc | `start b12 m128 seconds` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.b12.m128.mib}}` | calc | `start b12 m128 mib` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.b12.m128.peak}}` | calc | `start b12 m128 peak` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.m16.off1.ready}}` | calc | `start m16 off1 ready` | text | the last row for that start in starts.jsonl |
| `{{FIG:start.m16.off1.seconds}}` | calc | `start m16 off1 seconds` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.m16.off1.mib}}` | calc | `start m16 off1 mib` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.m16.off1.peak}}` | calc | `start m16 off1 peak` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.m16.on.ready}}` | calc | `start m16 on{m16} ready` | text | the last row for that start in starts.jsonl |
| `{{FIG:start.m16.on.seconds}}` | calc | `start m16 on{m16} seconds` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.m16.on.mib}}` | calc | `start m16 on{m16} mib` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.m16.on.peak}}` | calc | `start m16 on{m16} peak` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.m16.m128.ready}}` | calc | `start m16 m128 ready` | text | the last row for that start in starts.jsonl |
| `{{FIG:start.m16.m128.seconds}}` | calc | `start m16 m128 seconds` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.m16.m128.mib}}` | calc | `start m16 m128 mib` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.m16.m128.peak}}` | calc | `start m16 m128 peak` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.b31.off1.ready}}` | calc | `start b31 off1 ready` | text | the last row for that start in starts.jsonl |
| `{{FIG:start.b31.off1.seconds}}` | calc | `start b31 off1 seconds` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.b31.off1.mib}}` | calc | `start b31 off1 mib` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.b31.off1.peak}}` | calc | `start b31 off1 peak` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.b31.on280.ready}}` | calc | `start b31 on280 ready` | text | the last row for that start in starts.jsonl |
| `{{FIG:start.b31.on280.seconds}}` | calc | `start b31 on280 seconds` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.b31.on280.mib}}` | calc | `start b31 on280 mib` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.b31.on280.peak}}` | calc | `start b31 on280 peak` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.b31.on1120.ready}}` | calc | `start b31 on1120 ready` | text | the last row for that start in starts.jsonl |
| `{{FIG:start.b31.on1120.seconds}}` | calc | `start b31 on1120 seconds` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.b31.on1120.mib}}` | calc | `start b31 on1120 mib` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.b31.on1120.peak}}` | calc | `start b31 on1120 peak` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.b31.m128.ready}}` | calc | `start b31 m128 ready` | text | the last row for that start in starts.jsonl |
| `{{FIG:start.b31.m128.seconds}}` | calc | `start b31 m128 seconds` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.b31.m128.mib}}` | calc | `start b31 m128 mib` | n | the last row for that start in starts.jsonl |
| `{{FIG:start.b31.m128.peak}}` | calc | `start b31 m128 peak` | n | the last row for that start in starts.jsonl |
| `{{FIG:lat.b12.280.1}}` | calc | `lat m/b12/on280/latency.json 1` | pair | median / 95th percentile wall ms, status 200, over the requests of the arm |
| `{{FIG:lat.b12.280.4}}` | calc | `lat m/b12/on280/latency.json 4` | pair | median / 95th percentile wall ms, status 200, over the requests of the arm |
| `{{FIG:lat.b12.280.16}}` | calc | `lat m/b12/on280/latency.json 16` | pair | median / 95th percentile wall ms, status 200, over the requests of the arm |
| `{{FIG:lat.b12.280.tok}}` | calc | `lattok m/b12/on280/latency.json` | n | tokens(4 images) minus tokens(1), divided by 3 |
| `{{FIG:lat.b12.1120.1}}` | calc | `lat m/b12/on1120/latency.json 1` | pair | median / 95th percentile wall ms, status 200, over the requests of the arm |
| `{{FIG:lat.b12.1120.4}}` | calc | `lat m/b12/on1120/latency.json 4` | pair | median / 95th percentile wall ms, status 200, over the requests of the arm |
| `{{FIG:lat.b12.1120.16}}` | calc | `lat m/b12/on1120/latency.json 16` | pair | median / 95th percentile wall ms, status 200, over the requests of the arm |
| `{{FIG:lat.b12.1120.tok}}` | calc | `lattok m/b12/on1120/latency.json` | n | tokens(4 images) minus tokens(1), divided by 3 |
| `{{FIG:lat.m16.sel.1}}` | calc | `lat m/m16/on{m16}/latency.json 1` | pair | median / 95th percentile wall ms, status 200, over the requests of the arm |
| `{{FIG:lat.m16.sel.4}}` | calc | `lat m/m16/on{m16}/latency.json 4` | pair | median / 95th percentile wall ms, status 200, over the requests of the arm |
| `{{FIG:lat.m16.sel.16}}` | calc | `lat m/m16/on{m16}/latency.json 16` | pair | median / 95th percentile wall ms, status 200, over the requests of the arm |
| `{{FIG:lat.m16.sel.tok}}` | calc | `lattok m/m16/on{m16}/latency.json` | n | tokens(4 images) minus tokens(1), divided by 3 |
| `{{FIG:lat.b31.280.1}}` | calc | `lat m/b31/on280/latency.json 1` | pair | median / 95th percentile wall ms, status 200, over the requests of the arm |
| `{{FIG:lat.b31.280.4}}` | calc | `lat m/b31/on280/latency.json 4` | pair | median / 95th percentile wall ms, status 200, over the requests of the arm |
| `{{FIG:lat.b31.280.16}}` | calc | `lat m/b31/on280/latency.json 16` | pair | median / 95th percentile wall ms, status 200, over the requests of the arm |
| `{{FIG:lat.b31.280.tok}}` | calc | `lattok m/b31/on280/latency.json` | n | tokens(4 images) minus tokens(1), divided by 3 |
| `{{FIG:lat.b31.1120.1}}` | calc | `lat m/b31/on1120/latency.json 1` | pair | median / 95th percentile wall ms, status 200, over the requests of the arm |
| `{{FIG:lat.b31.1120.4}}` | calc | `lat m/b31/on1120/latency.json 4` | pair | median / 95th percentile wall ms, status 200, over the requests of the arm |
| `{{FIG:lat.b31.1120.16}}` | calc | `lat m/b31/on1120/latency.json 16` | pair | median / 95th percentile wall ms, status 200, over the requests of the arm |
| `{{FIG:lat.b31.1120.tok}}` | calc | `lattok m/b31/on1120/latency.json` | n | tokens(4 images) minus tokens(1), divided by 3 |
| `{{FIG:lat.b12.280.box}}` | calc | `latbox m/b12/on280/lat2.json` | n | median wall ms of the Box requests, status 200 |
| `{{FIG:lat.b31.280.box}}` | calc | `latbox m/b31/on280/lat2.json` | n | median wall ms of the Box requests, status 200 |
| `{{FIG:prefix.b12.280}}` | calc | `prefix m/b12/on280/prefix.json` | text | other question read from the cache of n; largest probability difference first read vs again |
| `{{FIG:fitT.b12.280.T}}` | file | `m/b12/on280/fitT.json > T` | r3 |  |
| `{{FIG:fitT.b12.280.cb}}` | file | `m/b12/on280/fitT.json > calib_logloss_before` | r3 |  |
| `{{FIG:fitT.b12.280.ca}}` | file | `m/b12/on280/fitT.json > calib_logloss_after` | r3 |  |
| `{{FIG:fitT.b12.280.rb}}` | file | `m/b12/on280/fitT.json > report_logloss_before` | r3 |  |
| `{{FIG:fitT.b12.280.ra}}` | file | `m/b12/on280/fitT.json > report_logloss_after` | r3 |  |
| `{{FIG:fitT.b12.1120.T}}` | file | `m/b12/on1120/fitT.json > T` | r3 |  |
| `{{FIG:fitT.b12.1120.cb}}` | file | `m/b12/on1120/fitT.json > calib_logloss_before` | r3 |  |
| `{{FIG:fitT.b12.1120.ca}}` | file | `m/b12/on1120/fitT.json > calib_logloss_after` | r3 |  |
| `{{FIG:fitT.b12.1120.rb}}` | file | `m/b12/on1120/fitT.json > report_logloss_before` | r3 |  |
| `{{FIG:fitT.b12.1120.ra}}` | file | `m/b12/on1120/fitT.json > report_logloss_after` | r3 |  |
| `{{FIG:fitT.m16.sel.T}}` | file | `m/m16/on{m16}/fitT.json > T` | r3 |  |
| `{{FIG:fitT.m16.sel.cb}}` | file | `m/m16/on{m16}/fitT.json > calib_logloss_before` | r3 |  |
| `{{FIG:fitT.m16.sel.ca}}` | file | `m/m16/on{m16}/fitT.json > calib_logloss_after` | r3 |  |
| `{{FIG:fitT.m16.sel.rb}}` | file | `m/m16/on{m16}/fitT.json > report_logloss_before` | r3 |  |
| `{{FIG:fitT.m16.sel.ra}}` | file | `m/m16/on{m16}/fitT.json > report_logloss_after` | r3 |  |
| `{{FIG:fitT.b31.280.T}}` | file | `m/b31/on280/fitT.json > T` | r3 |  |
| `{{FIG:fitT.b31.280.cb}}` | file | `m/b31/on280/fitT.json > calib_logloss_before` | r3 |  |
| `{{FIG:fitT.b31.280.ca}}` | file | `m/b31/on280/fitT.json > calib_logloss_after` | r3 |  |
| `{{FIG:fitT.b31.280.rb}}` | file | `m/b31/on280/fitT.json > report_logloss_before` | r3 |  |
| `{{FIG:fitT.b31.280.ra}}` | file | `m/b31/on280/fitT.json > report_logloss_after` | r3 |  |
| `{{FIG:fitT.b31.1120.T}}` | file | `m/b31/on1120/fitT.json > T` | r3 |  |
| `{{FIG:fitT.b31.1120.cb}}` | file | `m/b31/on1120/fitT.json > calib_logloss_before` | r3 |  |
| `{{FIG:fitT.b31.1120.ca}}` | file | `m/b31/on1120/fitT.json > calib_logloss_after` | r3 |  |
| `{{FIG:fitT.b31.1120.rb}}` | file | `m/b31/on1120/fitT.json > report_logloss_before` | r3 |  |
| `{{FIG:fitT.b31.1120.ra}}` | file | `m/b31/on1120/fitT.json > report_logloss_after` | r3 |  |
| `{{FIG:tree.fast.mean}}` | report | `box_tree > fast_vs_exact_complete > mean` | r4 | complete trees only |
| `{{FIG:tree.fast.p99}}` | report | `box_tree > fast_vs_exact_complete > p99` | r4 | complete trees only |
| `{{FIG:tree.fast.max}}` | report | `box_tree > fast_vs_exact_complete > max` | r4 | complete trees only |
| `{{FIG:tree.greedy.mean}}` | report | `box_tree > greedy_vs_exact_complete > mean` | r4 | complete trees only |
| `{{FIG:tree.greedy.p99}}` | report | `box_tree > greedy_vs_exact_complete > p99` | r4 | complete trees only |
| `{{FIG:tree.greedy.max}}` | report | `box_tree > greedy_vs_exact_complete > max` | r4 | complete trees only |
| `{{FIG:tree.complete}}` | report | `box_tree > complete` | n |  |
| `{{FIG:tree.capped}}` | report | `box_tree > capped` | n |  |
| `{{FIG:q35.n.S_accuracy}}` | report | `q35-new-minus-legacy > S\|report\|accuracy > n` | n | not in a run without the Qwen stage |
| `{{FIG:q35.diff.S_accuracy}}` | report | `q35-new-minus-legacy > S\|report\|accuracy > diff` | sd3 | not in a run without the Qwen stage |
| `{{FIG:q35.lo.S_accuracy}}` | report | `q35-new-minus-legacy > S\|report\|accuracy > lo` | sd3 | not in a run without the Qwen stage |
| `{{FIG:q35.hi.S_accuracy}}` | report | `q35-new-minus-legacy > S\|report\|accuracy > hi` | sd3 | not in a run without the Qwen stage |
| `{{FIG:q35.n.I_accuracy}}` | report | `q35-new-minus-legacy > I\|report\|accuracy > n` | n | not in a run without the Qwen stage |
| `{{FIG:q35.diff.I_accuracy}}` | report | `q35-new-minus-legacy > I\|report\|accuracy > diff` | sd3 | not in a run without the Qwen stage |
| `{{FIG:q35.lo.I_accuracy}}` | report | `q35-new-minus-legacy > I\|report\|accuracy > lo` | sd3 | not in a run without the Qwen stage |
| `{{FIG:q35.hi.I_accuracy}}` | report | `q35-new-minus-legacy > I\|report\|accuracy > hi` | sd3 | not in a run without the Qwen stage |
| `{{FIG:q35.n.S_logloss}}` | report | `q35-new-minus-legacy > S\|report\|logloss > n` | n | not in a run without the Qwen stage |
| `{{FIG:q35.diff.S_logloss}}` | report | `q35-new-minus-legacy > S\|report\|logloss > diff` | sd3 | not in a run without the Qwen stage |
| `{{FIG:q35.lo.S_logloss}}` | report | `q35-new-minus-legacy > S\|report\|logloss > lo` | sd3 | not in a run without the Qwen stage |
| `{{FIG:q35.hi.S_logloss}}` | report | `q35-new-minus-legacy > S\|report\|logloss > hi` | sd3 | not in a run without the Qwen stage |
| `{{FIG:q35.n.I_logloss}}` | report | `q35-new-minus-legacy > I\|report\|logloss > n` | n | not in a run without the Qwen stage |
| `{{FIG:q35.diff.I_logloss}}` | report | `q35-new-minus-legacy > I\|report\|logloss > diff` | sd3 | not in a run without the Qwen stage |
| `{{FIG:q35.lo.I_logloss}}` | report | `q35-new-minus-legacy > I\|report\|logloss > lo` | sd3 | not in a run without the Qwen stage |
| `{{FIG:q35.hi.I_logloss}}` | report | `q35-new-minus-legacy > I\|report\|logloss > hi` | sd3 | not in a run without the Qwen stage |
| `{{FIG:q35.T.new}}` | report | `qwen_fitT/new > T` | r3 | not in a run without the Qwen stage |
| `{{FIG:q35.T.legacy}}` | report | `qwen_fitT/legacy > T` | r3 | not in a run without the Qwen stage |
| `{{FIG:ts.n}}` | report | `baseline > all\|all > n_items` | n |  |
| `{{FIG:ts.n.S.noul}}` | report | `baseline > S.noul\|all > noul > n` | n |  |
| `{{FIG:ts.S.noul.accuracy}}` | report | `baseline > S.noul\|all > noul > accuracy` | r3 |  |
| `{{FIG:tsc.b12.S.noul.accuracy}}` | report | `baseline_ids/b12/280 > S.noul\|all > noul > accuracy` | r3 | the model at budget 280 on the baseline ids |
| `{{FIG:tsc.b31.S.noul.accuracy}}` | report | `baseline_ids/b31/280 > S.noul\|all > noul > accuracy` | r3 | the model at budget 280 on the baseline ids |
| `{{FIG:ts.n.S.choice}}` | report | `baseline > S.choice\|all > choice > n` | n |  |
| `{{FIG:ts.S.choice.accuracy}}` | report | `baseline > S.choice\|all > choice > accuracy` | r3 |  |
| `{{FIG:tsc.b12.S.choice.accuracy}}` | report | `baseline_ids/b12/280 > S.choice\|all > choice > accuracy` | r3 | the model at budget 280 on the baseline ids |
| `{{FIG:tsc.b31.S.choice.accuracy}}` | report | `baseline_ids/b31/280 > S.choice\|all > choice > accuracy` | r3 | the model at budget 280 on the baseline ids |
| `{{FIG:ts.n.S.score}}` | report | `baseline > S.score\|all > score > n` | n |  |
| `{{FIG:ts.S.score.accuracy}}` | report | `baseline > S.score\|all > score > accuracy` | r3 |  |
| `{{FIG:tsc.b12.S.score.accuracy}}` | report | `baseline_ids/b12/280 > S.score\|all > score > accuracy` | r3 | the model at budget 280 on the baseline ids |
| `{{FIG:tsc.b31.S.score.accuracy}}` | report | `baseline_ids/b31/280 > S.score\|all > score > accuracy` | r3 | the model at budget 280 on the baseline ids |
| `{{FIG:ts.n.R.noul}}` | report | `baseline > R.noul\|all > noul > n` | n |  |
| `{{FIG:ts.R.noul.accuracy}}` | report | `baseline > R.noul\|all > noul > accuracy` | r3 |  |
| `{{FIG:tsc.b12.R.noul.accuracy}}` | report | `baseline_ids/b12/280 > R.noul\|all > noul > accuracy` | r3 | the model at budget 280 on the baseline ids |
| `{{FIG:tsc.b31.R.noul.accuracy}}` | report | `baseline_ids/b31/280 > R.noul\|all > noul > accuracy` | r3 | the model at budget 280 on the baseline ids |
| `{{FIG:ts.n.R.choice}}` | report | `baseline > R.choice\|all > choice > n` | n |  |
| `{{FIG:ts.R.choice.accuracy}}` | report | `baseline > R.choice\|all > choice > accuracy` | r3 |  |
| `{{FIG:tsc.b12.R.choice.accuracy}}` | report | `baseline_ids/b12/280 > R.choice\|all > choice > accuracy` | r3 | the model at budget 280 on the baseline ids |
| `{{FIG:tsc.b31.R.choice.accuracy}}` | report | `baseline_ids/b31/280 > R.choice\|all > choice > accuracy` | r3 | the model at budget 280 on the baseline ids |
| `{{FIG:ts.n.S.box}}` | report | `baseline > S.box\|all > box > n` | n |  |
| `{{FIG:ts.S.box.mean_iou}}` | report | `baseline > S.box\|all > box > mean_iou` | r3 |  |
| `{{FIG:tsc.b12.S.box.mean_iou}}` | report | `baseline_ids/b12/280 > S.box\|all > box > mean_iou` | r3 | the model at budget 280 on the baseline ids |
| `{{FIG:tsc.b31.S.box.mean_iou}}` | report | `baseline_ids/b31/280 > S.box\|all > box > mean_iou` | r3 | the model at budget 280 on the baseline ids |
| `{{FIG:ts.S.box.iou50}}` | report | `baseline > S.box\|all > box > iou50` | r3 |  |
| `{{FIG:tsc.b12.S.box.iou50}}` | report | `baseline_ids/b12/280 > S.box\|all > box > iou50` | r3 | the model at budget 280 on the baseline ids |
| `{{FIG:tsc.b31.S.box.iou50}}` | report | `baseline_ids/b31/280 > S.box\|all > box > iou50` | r3 | the model at budget 280 on the baseline ids |
| `{{FIG:ts.S.box.conf_ece}}` | report | `baseline > S.box\|all > box > conf_ece` | r3 |  |
| `{{FIG:tsc.b12.S.box.conf_ece}}` | report | `baseline_ids/b12/280 > S.box\|all > box > conf_ece` | r3 | the model at budget 280 on the baseline ids |
| `{{FIG:tsc.b31.S.box.conf_ece}}` | report | `baseline_ids/b31/280 > S.box\|all > box > conf_ece` | r3 | the model at budget 280 on the baseline ids |
| `{{FIG:ts.S.box.absent_auroc}}` | report | `baseline > S.box\|all > box > absent_auroc` | r3 |  |
| `{{FIG:tsc.b12.S.box.absent_auroc}}` | report | `baseline_ids/b12/280 > S.box\|all > box > absent_auroc` | r3 | the model at budget 280 on the baseline ids |
| `{{FIG:tsc.b31.S.box.absent_auroc}}` | report | `baseline_ids/b31/280 > S.box\|all > box > absent_auroc` | r3 | the model at budget 280 on the baseline ids |
| `{{FIG:ts.n.R.box}}` | report | `baseline > R.box\|all > box > n` | n |  |
| `{{FIG:ts.R.box.mean_iou}}` | report | `baseline > R.box\|all > box > mean_iou` | r3 |  |
| `{{FIG:tsc.b12.R.box.mean_iou}}` | report | `baseline_ids/b12/280 > R.box\|all > box > mean_iou` | r3 | the model at budget 280 on the baseline ids |
| `{{FIG:tsc.b31.R.box.mean_iou}}` | report | `baseline_ids/b31/280 > R.box\|all > box > mean_iou` | r3 | the model at budget 280 on the baseline ids |
| `{{FIG:ts.R.box.iou50}}` | report | `baseline > R.box\|all > box > iou50` | r3 |  |
| `{{FIG:tsc.b12.R.box.iou50}}` | report | `baseline_ids/b12/280 > R.box\|all > box > iou50` | r3 | the model at budget 280 on the baseline ids |
| `{{FIG:tsc.b31.R.box.iou50}}` | report | `baseline_ids/b31/280 > R.box\|all > box > iou50` | r3 | the model at budget 280 on the baseline ids |
| `{{FIG:pair.gate_verdict}}` | calc | `gate gate_m16_vs_b12/{m16}` | text | point and interval readings of the five-measure gate |
