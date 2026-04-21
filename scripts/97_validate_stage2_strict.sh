#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${TRACE_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_SH="${CONDA_SH:-/home/changtian/miniconda3/etc/profile.d/conda.sh}"
TRACE_ENV="${TRACE_RUNNER_ENV:-trace-runner}"

source "$CONDA_SH"
conda activate "$TRACE_ENV"

cd "$ROOT"
mkdir -p results/logs

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="results/logs/stage2_strict_${STAMP}"
mkdir -p "$LOGDIR"

SUMMARY="$LOGDIR/summary.tsv"
touch "$SUMMARY"

echo "[TRACE] Stage 2 strict validation"
echo "[TRACE] Root: $ROOT"
echo "[TRACE] Log directory: $LOGDIR"

FAILED=0
STEP_NO=0

record_result() {
  local name="$1"
  local status="$2"
  local log="$3"
  printf "%s\t%s\t%s\n" "$name" "$status" "$log" >> "$SUMMARY"
}

run_step() {
  local name="$1"
  shift
  STEP_NO=$((STEP_NO + 1))
  local log="$LOGDIR/$(printf "%02d" "$STEP_NO")_${name}.log"

  echo
  echo "[TRACE] >>> $name"
  echo "[TRACE] Log: $log"
  echo "[TRACE] Command: $*"

  if "$@" > "$log" 2>&1; then
    echo "[TRACE] PASS: $name"
    record_result "$name" "PASS" "$log"
  else
    echo "[TRACE] FAIL: $name"
    echo "[TRACE] Last lines:"
    tail -n 80 "$log" || true
    record_result "$name" "FAIL" "$log"
    FAILED=$((FAILED + 1))
  fi
}

make_static_check() {
  cat > "$LOGDIR/static_checks.py" <<'PY'
from pathlib import Path
import sys

root = Path.cwd()
errors = []

train_patterns = [
    "src/pipeline/train",
    "src\\pipeline\\train",
    "src.pipeline.train",
]

scan_paths = [
    root / "scripts",
    root / "src" / "pipeline",
    root / "configs",
    root / "README.md",
]

for base in scan_paths:
    if not base.exists():
        continue
    paths = [base] if base.is_file() else list(base.rglob("*"))
    for path in paths:
        if not path.is_file():
            continue
        if path.suffix not in {".py", ".yaml", ".yml", ".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for pat in train_patterns:
            if pat in text:
                errors.append(f"{path.relative_to(root)} contains legacy pipeline reference: {pat}")

path_targets = [
    root / "src" / "cleaning" / "SCAREd" / "scared.py",
    root / "src" / "cleaning" / "Unified" / "Unified.py",
    root / "src" / "cleaning" / "horizon" / "horizon.py",
    root / "src" / "cleaning" / "BigDansing_Holistic" / "bigdansing.py",
    root / "src" / "cleaning" / "BigDansing_Holistic" / "holistic.py",
    root / "src" / "cleaning" / "BoostClean" / "activedetect" / "experiments" / "Experiment.py",
    root / "src" / "cleaning" / "holoclean-master" / "holoclean_run.py",
    root / "src" / "pipeline" / "cleaning_runner.py",
]

bad_patterns = [
    "/home/changtian/Cleaning-Clustering",
    "Repaired_res/Unified",
    '"./Repaired_res',
    "'./Repaired_res",
]

for path in path_targets:
    if not path.exists():
        errors.append(f"missing expected wrapper file: {path.relative_to(root)}")
        continue
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    for pat in bad_patterns:
        if pat in text:
            errors.append(f"{path.relative_to(root)} contains bad legacy path pattern: {pat}")

if errors:
    print("STATIC CHECK FAILED")
    for err in errors:
        print(err)
    sys.exit(1)

print("STATIC CHECK PASSED")
PY
}

make_boostclean_import_probe() {
  cat > "$LOGDIR/boostclean_import_probe.py" <<'PY'
import sys
import importlib
from pathlib import Path

root = Path.cwd() / "src" / "cleaning" / "BoostClean"
sys.path.insert(0, str(root.resolve()))

modules = [
    "activedetect",
    "activedetect.experiments.Experiment",
    "activedetect.reporting.CSVLogging",
    "activedetect.loaders.csv_loader",
    "activedetect.learning.BoostClean",
    "activedetect.learning.CleanClassifier",
    "activedetect.learning.EvaluateCleaning",
    "activedetect.error_detectors.QuantitativeErrorModule",
    "activedetect.error_detectors.PuncErrorModule",
    "activedetect.error_detectors.PatternErrorFinder",
]

failed = []
for module in modules:
    try:
        importlib.import_module(module)
        print(module, "OK")
    except Exception as exc:
        print(module, "FAILED:", repr(exc))
        failed.append(module)

raise SystemExit(1 if failed else 0)
PY
}

make_holoclean_import_probe() {
  cat > "$LOGDIR/holoclean_import_probe.py" <<'PY'
import sys
import importlib
from pathlib import Path

root = Path.cwd() / "src" / "cleaning" / "holoclean-master"
sys.path.insert(0, str(root.resolve()))

modules = [
    "holoclean",
    "dataset.dataset",
    "dataset.dbengine",
    "dataset.table",
    "dcparser.dcparser",
    "detect.detect",
    "detect.detector",
    "detect.violationdetector",
    "domain.domain",
    "repair.featurize",
]

failed = []
for module in modules:
    try:
        importlib.import_module(module)
        print(module, "OK")
    except Exception as exc:
        print(module, "FAILED:", repr(exc))
        failed.append(module)

raise SystemExit(1 if failed else 0)
PY
}

make_holoclean_db_check() {
  cat > "$LOGDIR/holoclean_db_check.py" <<'PY'
import os
import psycopg2

host = os.environ.get("TRACE_HOLOCLEAN_DB_HOST", "localhost")
port = int(os.environ.get("TRACE_HOLOCLEAN_DB_PORT", "5432"))
dbname = os.environ.get("TRACE_HOLOCLEAN_DB_NAME", "holo")
user = os.environ.get("TRACE_HOLOCLEAN_DB_USER", "holocleanuser")
password = os.environ.get("TRACE_HOLOCLEAN_DB_PASSWORD", "abcd1234")

print("host:", host)
print("port:", port)
print("dbname:", dbname)
print("user:", user)

conn = psycopg2.connect(
    host=host,
    port=port,
    dbname=dbname,
    user=user,
    password=password,
)
conn.close()
print("PostgreSQL connection: OK")
PY
}

assert_pipeline_manifest() {
  local expected_cleaned="$1"
  local expected_clustered="$2"
  local name="$3"

  python - "$expected_cleaned" "$expected_clustered" "$name" <<'PY'
import json
import sys
from pathlib import Path

expected_cleaned = int(sys.argv[1])
expected_clustered = int(sys.argv[2])
name = sys.argv[3]

p = Path("results/logs/pipeline_run_manifest.json")
if not p.exists():
    print(f"{name}: missing pipeline_run_manifest.json")
    sys.exit(1)

x = json.load(open(p, encoding="utf-8-sig"))
failure_count = int(x.get("failure_count", -1))
cleaned = int(x.get("cleaned_result_count", -1))
clustered = int(x.get("clustered_result_count", -1))

print(f"{name}: failure_count={failure_count}, cleaned={cleaned}, clustered={clustered}")

if failure_count != 0:
    sys.exit(1)
if cleaned < expected_cleaned:
    sys.exit(1)
if clustered < expected_clustered:
    sys.exit(1)
PY
}

assert_torch110_probe() {
  python - <<'PY'
import json
import sys

p = "results/logs/torch110_dependency_probe.json"
x = json.load(open(p, encoding="utf-8"))
missing = x["aggregate"]["missing_packages"]
print("torch110 missing_packages:", missing)
raise SystemExit(0 if not missing else 1)
PY
}

assert_clusterer_coverage() {
  python - <<'PY'
import json
import sys
from pathlib import Path

p = Path("results/logs/clusterer_coverage_summary.json")
if not p.exists():
    print("missing clusterer_coverage_summary.json")
    sys.exit(1)

x = json.load(open(p, encoding="utf-8-sig"))
manifest = x["outputs"]["pipeline_run_manifest.json"]
clusterers = x["outputs"]["clusterers"]

expected = ["HC", "DBSCAN", "GMM", "KMEANS", "KMEANSNF", "KMEANSPPS"]

print("manifest:", manifest)
print("clusterers:", clusterers)

if int(manifest.get("failure_count", -1)) != 0:
    sys.exit(1)
if int(manifest.get("clustered_result_count", -1)) < len(expected):
    sys.exit(1)

for name in expected:
    info = clusterers.get(name)
    if not info:
        print("missing clusterer:", name)
        sys.exit(1)
    if not info.get("observed_in_clustered_results"):
        print("clusterer not observed:", name)
        sys.exit(1)
    if not info.get("output_dir_exists"):
        print("clusterer output dir missing:", name)
        sys.exit(1)
PY
}

assert_cleaner_coverage() {
  local cleaner="$1"

  python - "$cleaner" <<'PY'
import json
import sys
from pathlib import Path

cleaner = sys.argv[1]
p = Path("results/logs/cleaner_coverage_summary.json")
if not p.exists():
    print(f"{cleaner}: missing cleaner_coverage_summary.json")
    sys.exit(1)

x = json.load(open(p, encoding="utf-8-sig"))
manifest = x["outputs"]["pipeline_run_manifest.json"]
cleaners = x["outputs"]["cleaners"]

print("manifest:", manifest)
print("cleaners:", cleaners)

if int(manifest.get("failure_count", -1)) != 0:
    sys.exit(1)
if int(manifest.get("cleaned_result_count", -1)) < 1:
    sys.exit(1)
if int(manifest.get("clustered_result_count", -1)) < 1:
    sys.exit(1)

info = cleaners.get(cleaner)
if not info:
    print(f"{cleaner}: missing cleaner summary")
    sys.exit(1)
if not info.get("observed_in_cleaned_results"):
    print(f"{cleaner}: not observed in cleaned_results")
    sys.exit(1)
if not info.get("cleaned_data_dir_exists"):
    print(f"{cleaner}: cleaned_data dir missing")
    sys.exit(1)
if not info.get("clustered_by", {}).get("HC"):
    print(f"{cleaner}: not clustered by HC")
    sys.exit(1)
PY
}

run_cleaner() {
  local cleaner="$1"
  STEP_NO=$((STEP_NO + 1))
  local log="$LOGDIR/$(printf "%02d" "$STEP_NO")_cleaner_${cleaner}.log"
  local timeout_value="${STAGE2_CLEANER_TIMEOUT:-8h}"

  echo
  echo "[TRACE] >>> cleaner_${cleaner}"
  echo "[TRACE] Log: $log"
  echo "[TRACE] Timeout: $timeout_value"

  if timeout "$timeout_value" python scripts/92_run_cleaner_coverage.py \
      --config configs/mode_b_smoke.yaml \
      --cleaners "$cleaner" \
      --clusterers HC \
      --cluster-trials "${STAGE2_CLUSTER_TRIALS:-5}" \
      --clean \
      --allow-failures \
      > "$log" 2>&1; then
    true
  else
    echo "[TRACE] Cleaner command returned non-zero: $cleaner"
  fi

  cp -f results/logs/cleaner_coverage_summary.json "$LOGDIR/cleaner_${cleaner}_summary.json" 2>/dev/null || true
  cp -f results/logs/pipeline_run_manifest.json "$LOGDIR/cleaner_${cleaner}_manifest.json" 2>/dev/null || true
  cp -f results/logs/pipeline_failures.json "$LOGDIR/cleaner_${cleaner}_failures.json" 2>/dev/null || true

  if assert_cleaner_coverage "$cleaner" >> "$log" 2>&1; then
    echo "[TRACE] PASS: cleaner_${cleaner}"
    record_result "cleaner_${cleaner}" "PASS" "$log"
  else
    echo "[TRACE] FAIL: cleaner_${cleaner}"
    echo "[TRACE] Last lines:"
    tail -n 120 "$log" || true
    record_result "cleaner_${cleaner}" "FAIL" "$log"
    FAILED=$((FAILED + 1))
  fi
}

make_static_check
make_boostclean_import_probe
make_holoclean_import_probe
make_holoclean_db_check

run_step "setup_mode_b" python scripts/00_setup_check.py --config configs/mode_b_smoke.yaml --strict
run_step "setup_mode_c" python scripts/00_setup_check.py --config configs/mode_c_full.yaml --check-all-data --strict
run_step "method_registry" python -m src.pipeline.method_registry --project-root . --show-disabled --check-paths
run_step "static_checks" python "$LOGDIR/static_checks.py"

run_step "mode_b_smoke" python scripts/90_run_smoke_from_scratch.py --config configs/mode_b_smoke.yaml --clean
cp -f results/logs/pipeline_run_manifest.json "$LOGDIR/mode_b_smoke_manifest.json" 2>/dev/null || true
if assert_pipeline_manifest 1 1 "mode_b_smoke" >> "$LOGDIR/mode_b_smoke_manifest_check.log" 2>&1; then
  record_result "mode_b_smoke_manifest" "PASS" "$LOGDIR/mode_b_smoke_manifest_check.log"
else
  record_result "mode_b_smoke_manifest" "FAIL" "$LOGDIR/mode_b_smoke_manifest_check.log"
  FAILED=$((FAILED + 1))
fi

run_step "clusterer_coverage" python scripts/91_run_clusterer_coverage.py --config configs/mode_b_smoke.yaml --clean
cp -f results/logs/clusterer_coverage_summary.json "$LOGDIR/clusterer_coverage_summary.json" 2>/dev/null || true
if assert_clusterer_coverage >> "$LOGDIR/clusterer_coverage_check.log" 2>&1; then
  record_result "clusterer_coverage_check" "PASS" "$LOGDIR/clusterer_coverage_check.log"
else
  record_result "clusterer_coverage_check" "FAIL" "$LOGDIR/clusterer_coverage_check.log"
  FAILED=$((FAILED + 1))
fi

run_step "torch110_dependency_probe" python scripts/93_probe_cleaner_dependencies.py --cleaners group:torch110 --conda-env torch110 --output results/logs/torch110_dependency_probe.json
if assert_torch110_probe >> "$LOGDIR/torch110_dependency_probe_check.log" 2>&1; then
  record_result "torch110_dependency_probe_check" "PASS" "$LOGDIR/torch110_dependency_probe_check.log"
else
  record_result "torch110_dependency_probe_check" "FAIL" "$LOGDIR/torch110_dependency_probe_check.log"
  FAILED=$((FAILED + 1))
fi

run_step "boostclean_import_probe" conda run -n activedetect python "$LOGDIR/boostclean_import_probe.py"
run_step "holoclean_import_probe" conda run -n hc37 python "$LOGDIR/holoclean_import_probe.py"
run_step "holoclean_db_check" conda run -n hc37 python "$LOGDIR/holoclean_db_check.py"

read -r -a CLEANERS <<< "${STAGE2_CLEANERS:-mode baran holoclean bigdansing boostclean horizon scared unified}"

for cleaner in "${CLEANERS[@]}"; do
  run_cleaner "$cleaner"
done

echo
echo "[TRACE] Stage 2 strict validation summary:"
cat "$SUMMARY"

if [ "$FAILED" -eq 0 ]; then
  echo "[TRACE] STAGE 2 STRICT VALIDATION PASSED"
  echo "PASSED" > "$LOGDIR/RESULT"
  exit 0
else
  echo "[TRACE] STAGE 2 STRICT VALIDATION FAILED: $FAILED failed step(s)"
  echo "FAILED" > "$LOGDIR/RESULT"
  exit 1
fi
