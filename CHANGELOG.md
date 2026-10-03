# Changelog

## [0.2.0](https://github.com/aminry/decisio/compare/v0.1.1...v0.2.0) (2026-10-03)


### ⚠ BREAKING CHANGES

* **serve:** no padding (--pad-policy none) is the MLX backend's default ([#46](https://github.com/aminry/decisio/issues/46))
* --multi-question sequential and --describe-options are the served defaults ([#38](https://github.com/aminry/decisio/issues/38))

### Features

* --describe-options shows an option's description alone when it has one (off by default) ([#36](https://github.com/aminry/decisio/issues/36)) ([f2d936e](https://github.com/aminry/decisio/commit/f2d936e5cbcdcd8f1880f1569121478e37b0c3ba))
* --multi-question sequential and --describe-options are the served defaults ([#38](https://github.com/aminry/decisio/issues/38)) ([06a1273](https://github.com/aminry/decisio/commit/06a1273fcb1668c856fd182008db26b81f90c1d4))
* --multi-question sequential makes each answer equal the question asked alone; stage timings and --pad-policy; README wording for multi-question requests ([#35](https://github.com/aminry/decisio/issues/35)) ([5461e0e](https://github.com/aminry/decisio/commit/5461e0e90e039a24c0426555d8801e003d81bc9c))
* **serve:** --noul-rendering words|letters|letters-keys (words by default) ([#44](https://github.com/aminry/decisio/issues/44)) ([597f383](https://github.com/aminry/decisio/commit/597f3838cdf0e4a7f89b87f2fd5a007f4e850ffe))
* **serve:** --noul-rendering words|letters|letters-keys, how a yes/no question is asked (words by default) ([597f383](https://github.com/aminry/decisio/commit/597f3838cdf0e4a7f89b87f2fd5a007f4e850ffe))
* **serve:** --pad-policy row, a single-question row padded to end on the block boundary (off by default) ([#45](https://github.com/aminry/decisio/issues/45)) ([8ccf143](https://github.com/aminry/decisio/commit/8ccf14367f13ff9ceb09726b2959072351873d03))
* **serve:** a cross-request prefix cache for the MLX engine (--prefix-cache-mb) ([#47](https://github.com/aminry/decisio/issues/47)) ([8e3f54f](https://github.com/aminry/decisio/commit/8e3f54f645faaf16329f7be2b317f1071cc9bf89))
* **serve:** an MLX engine for the text route on Apple silicon (--backend mlx) ([#39](https://github.com/aminry/decisio/issues/39)) ([f3c5026](https://github.com/aminry/decisio/commit/f3c5026dc03da3af71958d187a15e997a30d1d27))
* **serve:** no padding (--pad-policy none) is the MLX backend's default ([#46](https://github.com/aminry/decisio/issues/46)) ([2c1204e](https://github.com/aminry/decisio/commit/2c1204e4a6c0db22f1c95f20d0b5cfe1575b7a5b))


### Bug Fixes

* task registration on a certain readout, and the CPU stand-in's head engine (two HTTP 500s) ([#49](https://github.com/aminry/decisio/issues/49)) ([6e71298](https://github.com/aminry/decisio/commit/6e7129842ef9460ed35897d598e9b86affde6f2a))

## [0.1.1](https://github.com/aminry/decisio/compare/v0.1.0...v0.1.1) (2026-10-01)


### Bug Fixes

* **ci:** attest a CycloneDX SBOM of the image instead of SPDX ([#22](https://github.com/aminry/decisio/issues/22)) ([eef30b4](https://github.com/aminry/decisio/commit/eef30b4a8a99d82a3b1b14303ff1f9586a2b7a7e))

## [0.1.0](https://github.com/aminry/decisio/compare/v0.1.0...v0.1.0) (2026-10-01)


### Features

* a generator for the conformance items file from two public sets ([#18](https://github.com/aminry/decisio/issues/18)) ([77f1bbd](https://github.com/aminry/decisio/commit/77f1bbde3c23cf81d77bd38229df0a028a5efe71))


### Bug Fixes

* the head's output-layer rows load from a Hugging Face repo id as well as a local directory ([9d7d0ab](https://github.com/aminry/decisio/commit/9d7d0ab8133e682a336de8600a7588173c9a94dd))


### Documentation

* community files, CI and release tooling ([#2](https://github.com/aminry/decisio/issues/2)) ([4dc5388](https://github.com/aminry/decisio/commit/4dc5388eb5743f1ddc19845723d22ca0499e8fdb))
* launch checklist for the public flip ([#4](https://github.com/aminry/decisio/issues/4)) ([c79ec96](https://github.com/aminry/decisio/commit/c79ec9675bc2415a0ae5f045c2051a55022b2849))
* launch checklist gate for the image's first GPU start; README says it is pending ([dff92fb](https://github.com/aminry/decisio/commit/dff92fbe407336858ea26216a81c6ba0587aab96))
* launch checklist records the flip and corrects three steps ([#14](https://github.com/aminry/decisio/issues/14)) ([93a885d](https://github.com/aminry/decisio/commit/93a885dc82441aa77df39df304e0421d2e7ad7ac))
* launch checklist section for the maintainer's commit signing ([80e36ca](https://github.com/aminry/decisio/commit/80e36caa08c013d0a2fa11618baedc30695d5e35))
* README section Run with Docker; launch checklist covers the image ([2e93c65](https://github.com/aminry/decisio/commit/2e93c65f57cb4fc2f77b714666b593763b848616))
* record that the merge settings of checklist 2.1 are applied ([7b60810](https://github.com/aminry/decisio/commit/7b608109eed12618de516bfef6f56928c00fc4b5))
* record the history rebuild of main in the launch checklist ([b5688cc](https://github.com/aminry/decisio/commit/b5688cc3da89bc181855d1da2b6b10e5ece14ab8))
