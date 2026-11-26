# ZETA release maintenance

The repository has one coordinated version across harness/, infra/, cadence/ and website/. Shared 200ms scheduling code belongs in cadence/; Harness imports it instead of keeping copies.

Publish Harness, inference infrastructure and shared cadence implementation. Preserve the shipped serving/runtime source identities from infra/provenance/upstream-snapshot-files.json. The historical inventory also records research-only training/acceptance files omitted from this release. Keep internal evaluation pipelines, training loaders, model weights, credentials, raw trajectories and runtime state outside Git. Changes to frozen runtime require a new version. Source hashes and CPU tests do not establish GPU or live-game performance.

Run npm run check, npm test, python infra/tools/cpu_smoke.py and the native cadence Python tests for relevant changes. After final source edits, regenerate the integrated inventories and run npm run verify; retain the upstream inventory unchanged.
