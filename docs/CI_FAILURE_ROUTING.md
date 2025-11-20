# CI Failure Routing Guide

## Overview

This guide documents how CI failures are automatically routed to the appropriate teams based on exit codes from our evaluation and security scripts.

## Security Audit Exit Codes

The `scripts/security_audit.py` script returns specific exit codes for different security issue types, enabling precise CI/CD control and team routing.

### Exit Code Mapping

| Exit Code | Issue Type | Severity | Owner Team | Response Time | Auto-Page |
|-----------|------------|----------|------------|---------------|-----------|
| **0** | No issues found | - | - | - | No |
| **1** | General failure/mixed | Variable | DevOps Team | Next sprint | No |
| **2** | Critical vulnerabilities | **P0** | Security Team | **Immediate** | **Yes** |
| **3** | Exposed secrets | **P0** | Security Team | **Immediate** | **Yes** |
| **4** | High severity deps | P1 | Platform Team | Same day | No |
| **5** | Docker/container issues | P2 | Infrastructure Team | This week | No |
| **6** | Code security issues | P2 | Development Team | This sprint | No |

### Source Code Reference

See [`scripts/security_audit.py:418-453`](../scripts/security_audit.py#L418) for implementation:

```python
# Priority-based exit code assignment
if by_type.get("exposed_secret", 0) > 0:
    print("\n🔐 CRITICAL: Exposed secrets detected")
    exit_code = 3
elif by_severity.get("critical", 0) > 0:
    print("\n❌ CRITICAL: Critical vulnerabilities found")
    exit_code = 2
elif by_severity.get("high", 0) > 0 and args.fail_on_high:
    print("\n⚠️  HIGH: High severity issues found")
    exit_code = 4
# ... etc
```

## GitHub Actions Integration

### Basic CI Configuration

```yaml
# .github/workflows/security-check.yml
name: Security Audit with Routing

on:
  pull_request:
  push:
    branches: [main]

jobs:
  security-audit:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Run Security Audit
        id: security
        continue-on-error: true  # Capture exit code without failing job
        run: |
          python3 scripts/security_audit.py --severity high --fail-on-high
          echo "exit_code=$?" >> $GITHUB_OUTPUT

      - name: Route Based on Exit Code
        if: steps.security.outputs.exit_code != '0'
        uses: actions/github-script@v6
        with:
          script: |
            const exitCode = '${{ steps.security.outputs.exit_code }}';
            const routing = {
              '2': { team: '@org/security-team', severity: 'critical', page: true },
              '3': { team: '@org/security-team', severity: 'critical', page: true },
              '4': { team: '@org/platform-team', severity: 'high', page: false },
              '5': { team: '@org/infra-team', severity: 'medium', page: false },
              '6': { team: '@org/dev-team', severity: 'low', page: false }
            };

            const route = routing[exitCode] || { team: '@org/devops', severity: 'unknown' };

            // Create issue for tracking
            const issue = await github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: `🚨 Security Issue Detected (Exit Code: ${exitCode})`,
              body: `Security audit failed with exit code **${exitCode}**\n\n` +
                    `**Severity:** ${route.severity}\n` +
                    `**Team:** ${route.team}\n` +
                    `**Requires immediate attention:** ${route.page ? 'Yes' : 'No'}\n\n` +
                    `[View Workflow Run](${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId})`,
              labels: ['security', route.severity],
              assignees: route.team.replace('@org/', '').split('-')[0]  // Simplified
            });

            // If critical, also create alert
            if (route.page) {
              core.setFailed(`CRITICAL SECURITY ISSUE - Exit Code ${exitCode}`);
            }
```

## Slack Integration

### Slack Notification Setup

```yaml
      - name: Notify Slack
        if: steps.security.outputs.exit_code != '0'
        env:
          SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK }}
        run: |
          EXIT_CODE="${{ steps.security.outputs.exit_code }}"

          case "$EXIT_CODE" in
            2|3)
              CHANNEL="#security-critical"
              MENTION="<!channel>"
              COLOR="danger"
              ;;
            4)
              CHANNEL="#platform-alerts"
              MENTION="<!subteam^S12345678>"  # Platform team ID
              COLOR="warning"
              ;;
            5)
              CHANNEL="#infra-alerts"
              MENTION="@infra-oncall"
              COLOR="warning"
              ;;
            6)
              CHANNEL="#code-quality"
              MENTION=""
              COLOR="caution"
              ;;
            *)
              CHANNEL="#ci-failures"
              MENTION=""
              COLOR="caution"
              ;;
          esac

          curl -X POST $SLACK_WEBHOOK \
            -H 'Content-Type: application/json' \
            -d "{
              \"channel\": \"$CHANNEL\",
              \"attachments\": [{
                \"color\": \"$COLOR\",
                \"title\": \"Security Audit Failed (Exit Code: $EXIT_CODE)\",
                \"text\": \"$MENTION Security issues detected in ${{ github.repository }}\",
                \"fields\": [
                  {\"title\": \"Repository\", \"value\": \"${{ github.repository }}\", \"short\": true},
                  {\"title\": \"Branch\", \"value\": \"${{ github.ref }}\", \"short\": true},
                  {\"title\": \"Commit\", \"value\": \"${{ github.sha }}\", \"short\": true},
                  {\"title\": \"Exit Code\", \"value\": \"$EXIT_CODE\", \"short\": true}
                ],
                \"actions\": [{
                  \"type\": \"button\",
                  \"text\": \"View Run\",
                  \"url\": \"${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}\"
                }]
              }]
            }"
```

## PagerDuty Integration

### Critical Issues Auto-Page

```yaml
      - name: Page for Critical Issues
        if: steps.security.outputs.exit_code == '2' || steps.security.outputs.exit_code == '3'
        uses: actions/github-script@v6
        env:
          PAGERDUTY_TOKEN: ${{ secrets.PAGERDUTY_TOKEN }}
        with:
          script: |
            const exitCode = '${{ steps.security.outputs.exit_code }}';
            const severity = exitCode === '3' ? 'critical' : 'error';
            const title = exitCode === '3'
              ? 'EXPOSED SECRETS DETECTED'
              : 'CRITICAL VULNERABILITIES FOUND';

            const incident = {
              incident: {
                type: 'incident',
                title: `${title} in ${{ github.repository }}`,
                service: {
                  id: 'P12345',  // Your service ID
                  type: 'service_reference'
                },
                urgency: 'high',
                body: {
                  type: 'incident_body',
                  details: `Security audit exit code ${exitCode}\n` +
                          `Repository: ${{ github.repository }}\n` +
                          `Commit: ${{ github.sha }}\n` +
                          `Run: ${{ github.run_id }}`
                }
              }
            };

            await fetch('https://api.pagerduty.com/incidents', {
              method: 'POST',
              headers: {
                'Authorization': `Token token=${process.env.PAGERDUTY_TOKEN}`,
                'Content-Type': 'application/json',
                'Accept': 'application/vnd.pagerduty+json;version=2'
              },
              body: JSON.stringify(incident)
            });
```

## Team Configuration

### GitHub Teams Mapping

```yaml
# .github/CODEOWNERS
# Security Team - Critical issues (exit codes 2, 3)
/security/           @org/security-team
scripts/security_*   @org/security-team

# Platform Team - Dependencies (exit code 4)
package.json         @org/platform-team
requirements.txt     @org/platform-team
Pipfile             @org/platform-team

# Infrastructure Team - Docker/Container (exit code 5)
Dockerfile          @org/infra-team
docker-compose.yml  @org/infra-team
.dockerignore       @org/infra-team

# Development Team - Code quality (exit code 6)
/src/               @org/dev-team
/tests/             @org/dev-team
```

### Contact Matrix

| Team | GitHub Team | Slack Channel | Slack ID | PagerDuty Service | Email |
|------|-------------|---------------|----------|-------------------|-------|
| Security | @org/security-team | #security-critical | S_SECURITY | P_SEC_001 | security@company.com |
| Platform | @org/platform-team | #platform-alerts | S_PLATFORM | P_PLAT_001 | platform@company.com |
| Infrastructure | @org/infra-team | #infra-alerts | S_INFRA | P_INFRA_001 | infra@company.com |
| Development | @org/dev-team | #code-quality | S_DEV | - | dev@company.com |
| DevOps | @org/devops | #ci-failures | S_DEVOPS | P_OPS_001 | devops@company.com |

## Response Playbooks

### Exit Code 2: Critical Vulnerabilities

**Team:** Security Team
**Response Time:** Immediate
**Auto-Page:** Yes

1. **Immediate Actions:**
   - Security team member acknowledges page
   - Review vulnerability details in security audit report
   - Assess exploitability and exposure

2. **Mitigation:**
   - If actively exploitable: Consider emergency patch or rollback
   - Update affected dependencies immediately
   - Deploy hotfix if necessary

3. **Communication:**
   - Update incident channel with status
   - Notify affected service owners
   - Create post-mortem ticket

### Exit Code 3: Exposed Secrets

**Team:** Security Team
**Response Time:** Immediate
**Auto-Page:** Yes

1. **Immediate Actions:**
   - **CRITICAL:** Rotate exposed credentials immediately
   - Revoke compromised tokens/keys
   - Check audit logs for unauthorized access

2. **Investigation:**
   - Identify how secrets were exposed
   - Check git history for exposure duration
   - Review access logs for suspicious activity

3. **Remediation:**
   - Remove secrets from codebase (use BFG repo cleaner if needed)
   - Update secret management practices
   - Enable git-secrets hooks

### Exit Code 4: High Severity Dependencies

**Team:** Platform Team
**Response Time:** Same day
**Auto-Page:** No

1. **Assessment:**
   - Review dependency vulnerabilities
   - Check if vulnerable code paths are used
   - Evaluate update compatibility

2. **Resolution:**
   - Create PR with dependency updates
   - Run full test suite
   - Schedule deployment

### Exit Code 5: Docker/Container Issues

**Team:** Infrastructure Team
**Response Time:** This week
**Auto-Page:** No

1. **Review:**
   - Analyze container vulnerabilities
   - Check base image updates
   - Review security configurations

2. **Update:**
   - Update base images
   - Apply security best practices
   - Rebuild and test containers

### Exit Code 6: Code Security Issues

**Team:** Development Team
**Response Time:** This sprint
**Auto-Page:** No

1. **Triage:**
   - Review bandit/security findings
   - Categorize by risk level
   - Check for false positives

2. **Fix:**
   - Address high-risk issues first
   - Update code patterns
   - Add security tests

## Testing the Routing

### Manual Testing

```bash
# Test each exit code locally
for code in 0 1 2 3 4 5 6; do
  echo "Testing exit code $code..."

  # Simulate the exit code
  python3 -c "import sys; sys.exit($code)"
  RESULT=$?

  echo "Exit code: $RESULT should route to:"
  case $RESULT in
    0) echo "  ✅ Success - no routing" ;;
    1) echo "  📧 DevOps team (general failure)" ;;
    2) echo "  🚨 Security team (critical vulns) - PAGE" ;;
    3) echo "  🔐 Security team (exposed secrets) - PAGE" ;;
    4) echo "  ⚠️  Platform team (high deps)" ;;
    5) echo "  🐳 Infrastructure team (Docker)" ;;
    6) echo "  🔍 Development team (code security)" ;;
  esac
  echo
done
```

### CI Testing

```yaml
# .github/workflows/test-routing.yml
name: Test Exit Code Routing

on:
  workflow_dispatch:
    inputs:
      exit_code:
        description: 'Exit code to test (0-6)'
        required: true
        default: '0'

jobs:
  test-routing:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Simulate exit code
        run: |
          python3 -c "import sys; sys.exit(${{ github.event.inputs.exit_code }})"
        continue-on-error: true
        id: test

      - name: Display routing
        run: |
          echo "Exit code ${{ github.event.inputs.exit_code }} would route to:"
          # Add routing logic here
```

## Monitoring & Metrics

### Dashboard Queries

```sql
-- Exit code distribution (last 30 days)
SELECT
  exit_code,
  COUNT(*) as occurrences,
  MIN(timestamp) as first_seen,
  MAX(timestamp) as last_seen
FROM security_audit_runs
WHERE timestamp > NOW() - INTERVAL '30 days'
GROUP BY exit_code
ORDER BY exit_code;

-- Team response times
SELECT
  exit_code,
  team,
  AVG(time_to_acknowledge) as avg_ack_time,
  AVG(time_to_resolve) as avg_resolution_time
FROM incident_metrics
WHERE exit_code IN (2, 3, 4, 5, 6)
GROUP BY exit_code, team;
```

### Key Metrics

- **MTTA (Mean Time to Acknowledge)**: Target < 5 min for P0, < 1 hour for P1
- **MTTR (Mean Time to Resolve)**: Target < 1 hour for P0, < 1 day for P1
- **False Positive Rate**: Track and tune thresholds
- **Routing Accuracy**: Ensure issues go to right team first time

## Maintenance

### Quarterly Review

1. Review exit code distribution
2. Validate team assignments
3. Update response time SLAs
4. Tune severity thresholds

### Adding New Exit Codes

1. Update `scripts/security_audit.py` with new code
2. Document in this guide
3. Add routing logic to CI workflows
4. Update team notifications
5. Test routing before deploying

---

*Last Updated: 2025-11-19*
*Version: 1.0.0*
*Owner: Security & DevOps Teams*