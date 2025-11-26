---
name: Degraded Incident Report
about: Report and track a degraded (Faiss→memory) incident
title: "Degraded Incident: <date/time>"
labels: ops, degraded
assignees: 
---

## Summary
- Start time:
- End time (if recovered):
- Affected endpoints:

## Metrics
- similarity_degraded_total{event="degraded"}: 
- similarity_degraded_total{event="restored"}: 
- faiss_degraded_duration_seconds (peak/avg): 
- degradation_history_count (health): 

## Recovery
- Manual recovery attempts:
- Auto-recovery backoff settings:
- Next recovery ETA (if available):

## Logs/Events
- Key log excerpts:
- Related alerts fired:

## Root Cause (preliminary)

## Action Items
- [ ] Verify index path / availability
- [ ] Validate recovery loop configuration
- [ ] Update runbooks if needed

