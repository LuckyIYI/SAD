import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        try:
            import torch  # noqa: F401
        except Exception as exc:
            raise SystemExit(
                "PyTorch is required to build the SAD extension. "
                "If you're using pip, run with --no-build-isolation so the "
                "active torch install is visible."
            ) from exc

        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]

        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")
        if not cmake_generator or cmake_generator == "Ninja":
            try:
                import ninja  # noqa: F401

                ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                cmake_args += [
                    "-GNinja",
                    f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                ]
            except Exception:
                pass

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        subprocess.run(["cmake", ext.sourcedir, *cmake_args], cwd=build_temp, check=True)
        subprocess.run(["cmake", "--build", "."], cwd=build_temp, check=True)


setup(
    name="sad-ops",
    version="0.0.0",
    ext_modules=[CMakeExtension("sad_ops._sad_ext")],
    cmdclass={"build_ext": CMakeBuild},
    packages=find_packages(where="."),
    package_dir={"": "."},
    zip_safe=False,
    install_requires=["torch"],
    python_requires=">=3.9",
)
