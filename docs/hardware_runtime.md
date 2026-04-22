# Hardware and Runtime Notes

This document summarizes the observed runtime environment and practical hardware expectations.

## Maintainer Linux validation environment

Observed long-running validation was executed on a Linux server:

- Host label: `user-NF5280M6`
- CPU: 64-core class server
- Memory: approximately 1 TB
- Conda environment: `trace-runner`
- Additional method environments: `torch110`, `hc37`, `activedetect`
- PostgreSQL was used for HoloClean validation.

## Maintainer Windows development environment

Windows smoke/replay development was executed on:

- CPU: Intel i9-13900K class workstation
- Memory: 32 GB
- Python: 3.11

## Observed runtimes

Stage 2 strict validation passed on Linux. The observed wall-clock time was approximately 5 hours 47 minutes to 6 hours.

The longest steps are Baran and Unified. Low terminal output during these methods does not necessarily mean failure.

Mode B smoke validation is lightweight and typically finishes in seconds to minutes depending on the clusterer runtime.

## Minimum practical requirements

For quick Mode B smoke and Stage 3 replay:

- 4 CPU cores
- 8--16 GB RAM
- Python 3.11 or compatible environment
- No PostgreSQL required for the simplest smoke path

For full Stage 2 strict validation:

- Linux is recommended
- 16+ CPU cores recommended
- 64 GB RAM or more recommended
- Conda available in PATH
- PostgreSQL configured for HoloClean
- Several hours of wall-clock time

The full validation can run on smaller hardware, but reviewer runtime may be substantially longer.

