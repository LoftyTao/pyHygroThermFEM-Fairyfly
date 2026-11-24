# Source Provenance and Third-Party Notices

`pyHygroThermFEM-Fairyfly` builds its native C++ extension from vendored upstream sources: `vendor/HygroThermFEM` (`v_1.0.80` + 6 local patches, diff archived in [`patches/hygrothermfem-v1.0.80.patch`](../patches/hygrothermfem-v1.0.80.patch)) is vendored directly inside this repository, while `KeffCavity`, `LBNLCPPCommon`, `Windows-CalcEngine`, and `Eigen` remain unmodified upstream Git submodules.

| Component | Upstream source | Pinned tag / state | Integration |
|---|---|---|---|
| HygroThermFEM | [LBNL-ETA/HygroThermFEM](https://github.com/LBNL-ETA/HygroThermFEM) | `v_1.0.80` + 6 local patches | Vendored directly in `vendor/HygroThermFEM` (`99d94210` over `5ed842b6`) |
| KeffCavity | [LBNL-ETA/KeffCavity](https://github.com/LBNL-ETA/KeffCavity) | `Version_1.1.10` (`0a723fe3`) | Git submodule |
| LBNLCPPCommon | [LBNL-ETA/LBNLCPPCommon](https://github.com/LBNL-ETA/LBNLCPPCommon) | `v0.19` (`fe91f0e1`) | Git submodule |
| Windows-CalcEngine | [LBNL-ETA/Windows-CalcEngine](https://github.com/LBNL-ETA/Windows-CalcEngine) | `Version_1.0.78` (`e547fb82`) | Git submodule |
| Eigen | [libeigen/eigen](https://gitlab.com/libeigen/eigen) | `5.0.1` (`bc3b3987`) | Git submodule |

## Local patches applied in `vendor/HygroThermFEM` (relative to `v_1.0.80`)

1. `4e61760`: Add `ISO15099Ventilated` cavity mode and optional cavity height, pressure, and side emissivities.
2. `954af8a`: Pass `timestepIndex` to `boundaryHeatRates`.
3. `31ef1e0`: Index transient BC tables cyclically by declared period.
4. `c896fba`: Index periodic BC coefficients cyclically per field via `selectTimestep`.
5. `cae1455`: Accumulate per-step boundary exchange over accepted substeps (`m_PendingBoundaryExchange`).
6. `99d9421`: Restore 2x2 Gauss-point flux extrapolation to corner nodes in `IElementLinear2D::flux` and `fluxAtGaussPoints`.

The public snapshot omits the upstream HAMSTAD WP2 data and the validation cases that use or transcribe it.

## Runtime dependencies

- `libraryfemtherm==0.24.0` (LBNL BSD-3-Clause variant)
- `click>=8.1` (BSD-3-Clause; `colorama` on Windows is BSD-3-Clause)

## License files

Verbatim upstream license and copyright files are bundled under `pyhygrothermfem_fairyfly/third_party_licenses/` and summarized in [`third_party_notices.txt`](../pyhygrothermfem_fairyfly/third_party_notices.txt). `pyHygroThermFEM-Fairyfly`'s own wrapper and binding code is licensed under BSD-3-Clause (see [LICENSE](../LICENSE)); Eigen is primarily MPL-2.0 licensed with accompanying BSD, Apache, and MINPACK notices.

This exploratory repository is not affiliated with or endorsed by Lawrence Berkeley National Laboratory, the University of California, the U.S. Department of Energy, or Ladybug Tools, and makes no ISO, CEN, or NFRC standards certification claim.
