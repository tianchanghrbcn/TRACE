# Stage 2 Validation Summary

Maintainer-side Stage 2 strict validation passed on Linux.

The validation covered:

- setup check for Mode B,
- setup check for Mode C,
- method registry path check,
- static legacy-reference check,
- Mode B smoke run,
- clusterer coverage for HC, DBSCAN, GMM, KMEANS, KMEANSNF, and KMEANSPPS,
- torch110 dependency probe,
- BoostClean import probe,
- HoloClean import probe,
- HoloClean PostgreSQL connectivity check,
- individual cleaner coverage for mode,
- individual cleaner coverage for baran,
- individual cleaner coverage for holoclean,
- individual cleaner coverage for bigdansing,
- individual cleaner coverage for boostclean,
- individual cleaner coverage for horizon,
- individual cleaner coverage for scared,
- individual cleaner coverage for unified.

Observed result:

    PASSED

Observed maintainer runtime:

    Approximately 5 hours 47 minutes to 6 hours.

Reviewer hardware may produce different wall-clock times.

