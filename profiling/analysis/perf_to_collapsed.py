#!/usr/bin/env python3
# ABOUTME: Converts perf data to collapsed stacks

"""
Usage:
    collapsed-to-speedscope input.collapsed.txt [-o output.speedscope.json] [--name PROFILE_NAME]

The collapsed stack format is: semicolon;delimited;stack weight
Each unique first frame is treated as a separate thread/profile.

Examples:
    collapsed-to-speedscope perf_gcs_collapsed.txt
    collapsed-to-speedscope perf_gcs_collapsed.txt -o gcs.speedscope.json
    collapsed-to-speedscope perf_gcs_collapsed.txt --name "GCS Server"
"""

import argparse
import os
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Convert perf output to collapsed stack output"
    )
    parser.add_argument(
        "-i", "--input-dir", help="Directory containing perf files"
    )
    parser.add_argument(
        "-o", "--output-dir", help="Output path"
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        print(f"Error: {args.output} not found", file=sys.stderr)
        sys.exit(1)

    generate_collapsed_stacks(args.output_dir, args.input_dir)


def generate_collapsed_stacks(outdir, data_path=None):
    """Convert perf.data files to collapsed stack format.

    Produces *_collapsed.txt next to each *.data. Must be run on a machine
    whose kernel and runtime libraries match the one that recorded the data —
    for Ray worker profiles this means the worker node itself, before its
    /tmp/ray/session_* directory is torn down.

    Args:
        outdir: Directory to scan (ignored if data_path is given).
        data_path: If set, convert only this single .data file. Otherwise
                   convert every perf_*.data under outdir that does not
                   already have a matching _collapsed.txt.

    perf script output format:
        command  pid/tid  [cpu] timestamp: event:
            hex_addr func_name+0xoff (dso)
            hex_addr func_name+0xoff (dso)
            ...
        <blank line>
    """
    import glob as globmod

    if data_path is not None:
        data_files = [data_path]
    else:
        data_files = globmod.glob(os.path.join(outdir, "perf_*.data"))
    if not data_files:
        print("No perf data files found to convert.")
        return

    # Regex for the header line: "command pid/tid [cpu] timestamp: event:"
    header_re = re.compile(r"^\s*(\S+)\s+\d+")

    for data_path in data_files:
        name = os.path.basename(data_path).replace(".data", "")
        collapsed_path = os.path.join(
            os.path.dirname(data_path) or outdir, f"{name}_collapsed.txt"
        )
        if os.path.exists(collapsed_path):
            print(f"Skipping {data_path}: {collapsed_path} already exists")
            continue
        print(f"Converting {data_path} -> {collapsed_path}")
        try:
            perf_script = subprocess.Popen(
                ["perf", "script", "-i", data_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            with open(collapsed_path, "w") as out:
                comm = ""
                stack = []
                for line in perf_script.stdout:
                    line = line.decode("utf-8", errors="replace").rstrip()
                    if line == "":
                        if stack:
                            prefix = comm + ";" if comm else ""
                            out.write(prefix + ";".join(reversed(stack)) + " 1\n")
                            stack = []
                            comm = ""
                    elif line and line[0] in (" ", "\t"):
                        # Stack frame: leading whitespace + hex_addr + func_name+0xoff (dso)
                        parts = line.strip().split(" ", 1)
                        if len(parts) >= 2:
                            addr, rest = parts
                            # Split trailing "(dso)" off the end if present.
                            # Stripped libs (libcudart, libcublas, libtorch
                            # extensions, etc.) yield "[unknown] (dso)" — keep
                            # the DSO basename so per-library time is visible.
                            dso = ""
                            if rest.endswith(")"):
                                paren = rest.rfind(" (")
                                if paren != -1:
                                    dso = rest[paren + 2 : -1].strip()
                                    rest = rest[:paren]
                            func = _clean_func_name(rest)
                            if func == "[unknown]":
                                if dso and dso not in ("unknown", "[unknown]"):
                                    dso_base = os.path.basename(dso).replace(
                                        ";", ":"
                                    )
                                    # "@0x..." rather than "+0x..." so the
                                    # address survives collapsed_to_speedscope
                                    # (which strips trailing +0xHEX offsets).
                                    # analyze_pyspy_profile groups by DSO by
                                    # default and shows the @addr with
                                    # --detail-unknowns.
                                    func = f"unk_{dso_base}@0x{addr}"
                                else:
                                    func = f"unk_0x{addr}"
                            stack.append(func)
                    else:
                        # Header line: extract command name (thread)
                        m = header_re.match(line)
                        if m:
                            comm = m.group(1)
                if stack:
                    prefix = comm + ";" if comm else ""
                    out.write(prefix + ";".join(reversed(stack)) + " 1\n")
            perf_script.wait()
            size = os.path.getsize(collapsed_path)
            print(f"  -> {collapsed_path} ({size} bytes)")
        except Exception as e:
            print(f"  WARNING: Failed to convert {data_path}: {e}")


def _clean_func_name(name):
    """Strip hex offsets and clean up a function name from perf script output.

    Removes trailing +0x... offsets and replaces semicolons (which conflict
    with the collapsed stack delimiter) with colons.
    """
    # Strip trailing +0xHEX offset
    name = re.sub(r"\+0x[0-9a-f]+$", "", name)
    # Replace semicolons in C++ names (template args, etc.)
    name = name.replace(";", ":")
    return name


if __name__ == "__main__":
    main()
