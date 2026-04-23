#!/usr/bin/env python3
import os
import sys


def convert_metallib_to_header(metallib_path: str, header_path: str, target_name: str) -> None:
    with open(metallib_path, "rb") as f:
        data = f.read()

    header_content = (
        "// Auto-generated file containing embedded Metal library\n"
        "#pragma once\n"
        "#include <cstddef>\n"
        "#include <Metal/Metal.h>\n\n"
        f"namespace {target_name}_metal {{\n"
        "    static const unsigned char metallib_data[] = {\n"
    )

    bytes_per_line = 16
    for i in range(0, len(data), bytes_per_line):
        chunk = data[i : i + bytes_per_line]
        hex_values = ", ".join(f"0x{b:02x}" for b in chunk)
        header_content += f"        {hex_values},"
        if i + bytes_per_line < len(data):
            header_content += "\n"

    header_content += (
        "\n    };\n"
        f"    static const size_t metallib_data_len = {len(data)};\n\n"
        "    inline id<MTLLibrary> createLibrary(id<MTLDevice> device, NSError** error = nullptr) {\n"
        "        dispatch_data_t libraryData = dispatch_data_create(\n"
        "            metallib_data,\n"
        "            metallib_data_len,\n"
        "            dispatch_get_main_queue(),\n"
        "            ^{ /* no-op */ });\n\n"
        "        NSError* localError = nil;\n"
        "        id<MTLLibrary> library = [device newLibraryWithData:libraryData error:&localError];\n\n"
        "        if (error) {\n"
        "            *error = localError;\n"
        "        }\n\n"
        "        return library;\n"
        "    }\n"
        f"}} // namespace {target_name}_metal\n"
    )

    dir_path = os.path.dirname(header_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(header_path, "w") as f:
        f.write(header_content)

    print(f"Generated {header_path} ({len(data)} bytes)")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: metallib_to_header.py <metallib_path> <header_path> <target_name>")
        sys.exit(1)

    metallib_path = sys.argv[1]
    header_path = sys.argv[2]
    target_name = sys.argv[3]

    convert_metallib_to_header(metallib_path, header_path, target_name)
