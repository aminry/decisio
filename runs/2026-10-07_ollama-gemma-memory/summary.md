# The Gemma Ollama listing's memory (2026-10-07)

`ollama.com/aminroudaki/decisio-gemma` packages the Gemma 4 12B base for Ollama's decision route.
Each tag was loaded alone with Ollama 0.35.1 on an Apple M5 Pro (64 GB) and read from `/api/ps` at the listing's context (`num_ctx` 8192, set in its Modelfile), then unloaded.

| Tag | Download | Loaded (`size`, all in GPU memory) | Digest |
| --- | ---: | ---: | --- |
| `q4_k_m` (also `latest`) | 7.7 GB | 8.57 GB | `e14e68fe0e2c` |
| `q8_0` | 12 GB | 13.58 GB | `8352db2c9b1e` |

So the `q4_k_m` tag leaves about 7 GB of a 16 GB machine free, which makes this listing the Ollama path for 16 GB laptops.
It was measured on one 64 GB Mac; a 16 GB machine was not measured.
The tags carry no vision projector (the decision route does not need one), so these are below the 8.64 and 13.64 GB measured earlier on the `hf.co` pulls of the same GGUFs, which include it.

## Files

`api_ps_<tag>.json`: Ollama's `/api/ps` with that tag loaded; `host.txt`: the date, Ollama version and machine.
