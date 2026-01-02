# Security Runtime Verification: Token Rotation & Opcode Blocking

- Date: 2025-12-27
- Scope: Validate admin token rotation and model opcode blocking

## Commands
- .venv/bin/pytest tests/unit/test_admin_token_rotation.py -q
- .venv/bin/pytest tests/unit/test_model_opcode_modes.py -q

## Result
- PASS

## Notes
- Admin token rotation: old token rejected (403), new token accepted (200)
- Opcode blocking: blacklist/whitelist modes block GLOBAL opcode pickle payloads; audit mode records but does not block
