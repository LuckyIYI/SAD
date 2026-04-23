#!/usr/bin/env python3
"""Generate backends/webgpu_js/shaders.js from shared WGSL."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
WGSL_PATH = ROOT / "sad_shared.wgsl"
OUT_PATH = ROOT.parent / "webgpu_js" / "shaders.js"


def main() -> None:
    code = WGSL_PATH.read_text(encoding="utf-8")
    escaped = code.replace("`", "\\`").replace("${", "\\${")
    out = (
        "// Generated from backends/shared/sad_shared.wgsl.\n"
        "window.SAD_SHADER_CODE = `\n"
        f"{escaped}\n"
        "`;\n"
    )
    OUT_PATH.write_text(out, encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
