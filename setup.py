from pathlib import Path
import importlib.util
import json
import os
import shutil
import subprocess
import sys

import pybind11
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist


ROOT = Path(__file__).parent.resolve()
_source_spec = importlib.util.spec_from_file_location(
    'pyhygrotherm_build_source', ROOT / 'build_source.py')
_build_source = importlib.util.module_from_spec(_source_spec)
_source_spec.loader.exec_module(_build_source)


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def build_extension(self, extension):
        cmake = shutil.which("cmake")
        if cmake is None:
            raise RuntimeError("CMake is required to build pyHygroThermFEM-Fairyfly.")

        native_path = Path(self.get_ext_fullpath(extension.name)).resolve()
        output_dir = native_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        build_dir = Path(self.build_temp).resolve() / "_native"
        build_dir.mkdir(parents=True, exist_ok=True)
        configuration = "Debug" if self.debug else "Release"

        # Build the child environment from os.environ instead of letting the
        # subprocess inherit the raw environment block. A parent process can
        # hand us a block containing both 'PATH' and 'Path'; those keys are
        # distinct to CreateProcess but collide case-insensitively inside
        # .NET Framework's ProcessStartInfo, which makes MSBuild fail with
        # MSB6001 when it launches CL.exe.
        env = dict(os.environ)

        subprocess.run(
            [
                cmake,
                "-S",
                str(ROOT),
                "-B",
                str(build_dir),
                f"-DPython_EXECUTABLE={sys.executable}",
                f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_dir}",
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{configuration.upper()}={output_dir}",
                f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={output_dir}",
                f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{configuration.upper()}={output_dir}",
                f"-DCMAKE_BUILD_TYPE={configuration}",
            ],
            check=True,
            env=env,
        )
        subprocess.run(
            [
                cmake,
                "--build",
                str(build_dir),
                "--config",
                configuration,
                "--target",
                "_native",
            ],
            check=True,
            env=env,
        )

        identity_path = build_dir / 'build-source-identity.json'
        identity = json.loads(identity_path.read_text(encoding='utf-8'))
        if _build_source.prepare_source(ROOT) != identity:
            raise RuntimeError('Build inputs changed after CMake source verification.')
        # build_py has already run for wheels. Write beside the actual extension
        # in build_lib so install_lib copies this file into the wheel directly.
        _build_source.finalize_package(output_dir, identity_path, native_path)

    def get_outputs(self):
        outputs = super().get_outputs()
        for extension in self.extensions:
            metadata = str(Path(self.get_ext_fullpath(extension.name)).parent /
                           '_build_source.json')
            if metadata not in outputs:
                outputs.append(metadata)
        return outputs

    def copy_extensions_to_source(self):
        super().copy_extensions_to_source()
        build_py = self.get_finalized_command('build_py')
        for extension in self.extensions:
            package = self.get_ext_fullname(extension.name).rpartition('.')[0]
            metadata = Path(self.build_lib).joinpath(*package.split('.'),
                                                     '_build_source.json')
            destination = Path(build_py.get_package_dir(package)) / metadata.name
            self.copy_file(str(metadata), str(destination))


class VerifiedSdist(sdist):
    def make_release_tree(self, base_dir, files):
        proof, identity = _build_source._prepared_source(ROOT)
        # Include every verified input, including extra root/CMake controls of
        # any suffix, without relying on SCM discovery in a Git-free re-sdist.
        files = sorted(set(files).union(proof['files']))
        super().make_release_tree(base_dir, files)
        destination = Path(base_dir).resolve()
        if (ROOT / '.git').exists():
            frozen = _build_source.freeze_source(
                ROOT, identity['source_commit'], destination)
        else:
            # make_release_tree may already have hard-linked the proof. Do not
            # copy a file onto that same inode when re-packaging a source package.
            if not (destination / '_source_freeze.json').is_file():
                shutil.copyfile(ROOT / '_source_freeze.json',
                                destination / '_source_freeze.json')
            frozen = _build_source.prepare_source(destination)
        if frozen != identity:
            raise RuntimeError('Source package differs from verified build inputs.')


readme = ROOT / "README.md"
long_description = readme.read_text(encoding="utf-8") if readme.is_file() else ""
requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

setup(
    name="pyHygroThermFEM-Fairyfly",
    description="Python interface to HygroThermFEM for THMZ simulations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Repository": "https://github.com/LoftyTao/pyHygroThermFEM-Fairyfly",
        "Upstream HygroThermFEM": "https://github.com/LBNL-ETA/HygroThermFEM",
    },
    use_scm_version=True,
    packages=find_packages(exclude=("tests", "tests.*")),
    ext_modules=[CMakeExtension("pyhygrothermfem_fairyfly._native")],
    cmdclass={"build_ext": CMakeBuild, "sdist": VerifiedSdist},
    install_requires=requirements.splitlines(),
    include_package_data=False,
    package_data={
        'pyhygrothermfem_fairyfly': [
            '_build_source.json', 'third_party_notices.txt',
            'third_party_licenses/*/*',
        ]
    },
    entry_points={
        "console_scripts": [
            "pyhygrothermfem-fairyfly=pyhygrothermfem_fairyfly.cli:main"
        ]
    },
    python_requires=">=3.11,<3.16",
    license="BSD-3-Clause",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Python :: Implementation :: CPython",
        "Operating System :: Microsoft :: Windows",
    ],
    zip_safe=False,
)
