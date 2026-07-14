# stack traces are sampled using 'perf record' with call graphs
sudo perf record -g --call-graph fp -F 99 -p <pid> -o <data-path>

# perf_to_collapsed.py  is used to generate 'collapsed' stacks
profiling/analysis/perf_to_collapsed.py -o <output-dir>

# analyze_perf_profiles.sh converts collapsed stacks to speedscope format
profiling/analysis/analyze_perf_profiles.sh

# analyze_pyspy_profile.py can process the speedscope files, or use speedscope.app
# there are many options to get top funcs and callers of specific functions, but
# speedscope.app can generate flame graphs in the browser
analyze_pyspy_profile.py --list-threads <speedscope-json-path>
