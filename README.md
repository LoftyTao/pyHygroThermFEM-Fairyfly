English | [简体中文](README.zh-CN.md)

# pyHygroThermFEM-Fairyfly

`pyHygroThermFEM-Fairyfly` provides Windows x64 Python bindings (`pyhygrothermfem_fairyfly`, CPython 3.11–3.15) and a `run-thmz` CLI for [HygroThermFEM](https://github.com/LBNL-ETA/HygroThermFEM) and [KeffCavity](https://github.com/LBNL-ETA/KeffCavity) to solve premeshed THMZ files in an isolated solver environment. This repository is an exploratory prototype built for Ladybug Tools integration and easy distribution in Ladybug Tools MCP workflows, and is not planned for long-term maintenance because the HygroThermFEM maintainers are the right owners for official Python bindings.

The patched `HygroThermFEM` source (`v_1.0.80` with local fixes) is bundled directly under `vendor/HygroThermFEM`, while `KeffCavity`, `LBNLCPPCommon`, `Windows-CalcEngine`, and `Eigen` are linked as upstream Git submodules. It pairs with [fairyfly-next-therm](https://github.com/LoftyTao/fairyfly-next-therm) and [fairyfly-next-grasshopper](https://github.com/LoftyTao/fairyfly-next-grasshopper), and uses [LibraryFEMTherm](https://github.com/LBNL-ETA/LibraryFEMTherm) at runtime.

## Build and Usage

Building from source requires Visual Studio Build Tools with C++20 support, CMake, and Windows x64 CPython 3.11–3.15 (`3.13` recommended).

```powershell
git clone --recursive https://github.com/LoftyTao/pyHygroThermFEM-Fairyfly.git
cd pyHygroThermFEM-Fairyfly
python -I -m pip install -r dev-requirements.txt
python -I -m build --no-isolation --wheel
python -I -m pip install --force-reinstall dist\pyhygrothermfem_fairyfly-*.whl
python -I -m pyhygrothermfem_fairyfly run-thmz slab.thmz --output slab_solved.thmz
```

The wrapper and binding code in this repository is licensed under BSD-3-Clause, and all upstream C++ libraries retain their original licenses in the repository.
