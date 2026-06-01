# Telemak Operator Deploy / Rollback

Telemak keeps `v0.6.15` as the bronze rollback baseline:

- Git tag: `v0.6.15-stable`
- Remote rollback directory: `~/telemak/Release.bronze-0.6.15`

## Inventory

The public repository ships only `scripts/telemak-hosts.example.sh`.
Copy it to `scripts/telemak-hosts.local.sh` and edit the LAN inventory for
the operator site. The local file is gitignored.

Example host names:

- `node-a` -> `/Users/admin/telemak/Release`, port `8003`
- `node-b` -> `/Users/admin/telemak/Release`, port `8003`
- `large-node` -> `/Users/admin/telemak/Release.deepseek`, port `8013`

## Deploy All

```bash
scripts/deploy-all.sh --canary node-a
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
scripts/rollback-host.sh node-a
```

Rollback to the latest previous release:

```bash
scripts/rollback-host.sh node-a prev1
```

Rollback accepts inventory name, IP, or last octet:

```bash
scripts/rollback-host.sh 10.0.0.12 bronze-0.6.15
scripts/rollback-host.sh <telemak-host> prev1
```

The script refuses targets without `BRONZE-MANIFEST.txt` or a runnable
`telemak --version`, unless `--force` is passed.
