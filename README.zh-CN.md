[English](README.md) | 简体中文

# pyHygroThermFEM-Fairyfly

`pyHygroThermFEM-Fairyfly` 为 [HygroThermFEM](https://github.com/LBNL-ETA/HygroThermFEM) 与 [KeffCavity](https://github.com/LBNL-ETA/KeffCavity) 提供 Windows x64 Python 封装（`pyhygrothermfem_fairyfly`，支持 CPython 3.11–3.15）与 `run-thmz` 命令行工具，用于在独立隔离环境中求解预分网 THMZ 文件。本仓库仅作为面向 Ladybug Tools 生态与 Ladybug Tools MCP 易分发方案的探索原型，无长期维护计划，HygroThermFEM 官方团队才是官方 Python 绑定的最佳长期维护者。

仓库已直接在 `vendor/HygroThermFEM` 内打包带本地修复补丁的 `HygroThermFEM`（`v_1.0.80`）源码，并将 `KeffCavity`、`LBNLCPPCommon`、`Windows-CalcEngine` 与 `Eigen` 作为上游 Git 子模块引入。本库配合 [fairyfly-next-therm](https://github.com/LoftyTao/fairyfly-next-therm) 与 [fairyfly-next-grasshopper](https://github.com/LoftyTao/fairyfly-next-grasshopper) 使用，运行时依赖 [LibraryFEMTherm](https://github.com/LBNL-ETA/LibraryFEMTherm)。

## 构建与使用

从源码编译需要配备 C++20 桌面工具链的 Visual Studio Build Tools、CMake 以及 Windows x64 CPython 3.11–3.15（推荐 `3.13`）。

```powershell
git clone --recursive https://github.com/LoftyTao/pyHygroThermFEM-Fairyfly.git
cd pyHygroThermFEM-Fairyfly
python -I -m pip install -r dev-requirements.txt
python -I -m build --no-isolation --wheel
python -I -m pip install --force-reinstall dist\pyhygrothermfem_fairyfly-*.whl
python -I -m pyhygrothermfem_fairyfly run-thmz slab.thmz --output slab_solved.thmz
```

本仓库自有封装与绑定代码采用 BSD-3-Clause 协议，所有上游 C++ 依赖库均在仓库内完整保留其原始许可证声明。
