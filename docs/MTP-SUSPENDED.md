# MTP Suspended

As of 2026-05-27, native MTP work in Telemak is officially suspended.

Stable baseline:

- Telemak: `0.6.2`
- `mlx-swift-lm` fork branch: `feat/v2-mtp-ssm-rollback-pre-moe`
- fork commit: `179026a`

Reason:

- Gemma 4 26B-A4B already reaches about 72 tok/s without MTP on max-64.
- The Gemma MoE loader port (`ea0758f`) made the model loadable but caused
  catastrophic generation speed regressions.
- The expected MTP gain does not currently justify more implementation risk
  versus higher-priority production work.

Do not resume MTP implementation by default. Reopen it only as a fresh spike
with a proven external baseline on the same host, same model, and same prompt
suite.
