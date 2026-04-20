# Setup Scripts

This directory contains normalized setup entry points.

## Recommended order

### Mode A

```bash
bash scripts/setup/setup_mode_a.sh
$setupReadme = @'
# Setup Scripts

This directory contains normalized setup entry points.

## Recommended order

### Mode A

```bash
bash scripts/setup/setup_mode_a.sh

Creates or updates the lightweight trace-runner environment.

Mode B
bash scripts/setup/setup_mode_b.sh

Creates the smoke-test environment. It currently reuses trace-runner.

Mode C
bash scripts/setup/setup_mode_c_full.sh

Creates the original full pipeline environment and auxiliary environments.

Mode C may require system packages and database services for HoloClean and other cleaners.
It is intended for full from-scratch execution, not for the first artifact setup check.

Local system configuration

Copy:

cp configs/runtime/system.example.env configs/runtime/system.local.env

Then edit configs/runtime/system.local.env.

Do not commit system.local.env.