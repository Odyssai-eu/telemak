# Telemak Operator Deploy / Rollback

Telemak keeps `v0.6.15` as the bronze rollback baseline:

- Git tag: `v0.6.15-stable`
- Remote rollback directory: `~/telemak/Release.bronze-0.6.15`

## Inventory

The host inventory is centralized in `scripts/telemak-hosts.sh`.

Supported hosts:

- `ultra-512` / `.29` -> `/Users/admin/telemak/Release.deepseek`, port `8013`, LaunchAgent `eu.odyssai.telemak.deepseek`
- `ultra-256a` / `.30` -> `/Users/admin/telemak/Release`
- `ultra-256b` / `.31` -> `/Users/admin/telemak/Release`
- `ultra-256c` / `.32` -> `/Users/admin/telemak/Release`
- `ultra-96` / `.49` -> `/Users/admin/telemak/Release`
- `max-64` / `.50` -> `/Users/admin/telemak/Release`

## Deploy All

```bash
scripts/deploy-all.sh --canary ultra-256a
```

The script builds `Release`, deploys only the canary first, runs smoke checks,
then rolls out to every remaining host.

Smoke checks:

- `telemak --version`
- `GET /health`
- `GET /admin/activity`
- menubar LaunchAgent status when configured

If the canary fails, no other host is touched.

## Rollback One Host

Rollback to the bronze baseline:

```bash
scripts/rollback-host.sh ultra-256a
```

Rollback to the latest previous release:

```bash
scripts/rollback-host.sh ultra-256a prev1
```

Rollback accepts inventory name, IP, or last octet:

```bash
scripts/rollback-host.sh .32 bronze-0.6.15
scripts/rollback-host.sh 192.168.86.50 prev1
```

The script refuses targets without `BRONZE-MANIFEST.txt` or a runnable
`telemak --version`, unless `--force` is passed.
