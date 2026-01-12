# METRICS_DISABLED_GUARDS_TARGETED_TESTS_DESIGN

## Goal
- Confirm the new guard clauses skip metrics-dependent unit tests when metrics are disabled.

## Plan
- Run targeted pytest selections for the updated tests.
- Expect skip outcomes with a consistent reason message.
