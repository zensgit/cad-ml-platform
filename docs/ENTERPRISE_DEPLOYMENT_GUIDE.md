# 🏢 Enterprise Deployment Guide

**Version**: 1.0.0
**Date**: 2025-11-30
**Target Audience**: DevOps Engineers, Security Officers, System Administrators

## 1. Overview

This guide details the deployment of the CAD ML Platform in an enterprise environment, focusing on security, compliance, and high availability. It builds upon the standard [Deployment Guide](DEPLOYMENT_GUIDE_V7.md) but adds strict governance controls.

## 2. Security Architecture

### 2.1 Network Security
- **Ingress**: All traffic must pass through a WAF (Web Application Firewall).
- **TLS**: TLS 1.3 is mandatory for all external and internal communication.
- **Segmentation**:
  - `frontend` (Public Subnet)
  - `api-server` (Private Subnet)
  - `redis/db` (Data Subnet - No Internet Access)

### 2.2 Container Security
- **Base Images**: Use distroless or minimal Alpine images where possible.
- **User**: All containers must run as non-root user (UID > 1000).
- **Read-Only**: Root filesystems should be mounted read-only where feasible.

### 2.3 Secrets Management
- **Do NOT** use environment variables for sensitive secrets in plain text.
- **Production**: Use HashiCorp Vault, AWS Secrets Manager, or Kubernetes Secrets.
- **Rotation**: API Keys and Database credentials must be rotated every 90 days.

## 3. Compliance & Governance

### 3.1 Software Bill of Materials (SBOM)
Every deployment artifact must be accompanied by an SBOM.
```bash
# Generate SBOM before deployment
make generate-sbom
```
The SBOM (`sbom.json`) should be archived with the release artifacts.

### 3.2 License Compliance
Before any upgrade, verify third-party licenses:
```bash
# Fail if restricted licenses (e.g., GPL) are found
make check-licenses
```

### 3.3 Audit Logs
- **Federated Learning**: All participant updates are logged to `logs/federated_audit.log`.
- **Access Logs**: API access logs must be shipped to a central SIEM (e.g., Splunk, ELK).

## 4. Automated Deployment Pipeline

We provide a GitHub Actions workflow (`.github/workflows/enterprise_release.yml`) that enforces these checks.

### 4.1 Pipeline Stages
1. **Security Scan**: `bandit` (SAST) and `safety` (Dependencies).
2. **License Check**: Verifies compliance.
3. **SBOM Generation**: Creates inventory.
4. **Build & Sign**: Builds Docker image and signs it (Cosign).
5. **Deploy**: Pushes to staging/production.

### 4.2 Manual Gate
Production deployments require manual approval in the CI/CD system after Staging verification.

## 5. Monitoring & Incident Response

### 5.1 Key Metrics (Prometheus)
- `cad_ml_security_violations_total`: Counter for auth failures/WAF blocks.
- `cad_ml_compliance_status`: Gauge (1=Compliant, 0=Non-Compliant).

### 5.2 Alerting
- **Severity 1 (Page)**: API Availability < 99.9%, Security Breach detected.
- **Severity 2 (Ticket)**: License check warning, High latency.

## 6. Disaster Recovery

- **RPO (Recovery Point Objective)**: 1 hour (Database snapshots).
- **RTO (Recovery Time Objective)**: 4 hours.
- **Backup Strategy**:
  - Database: Hourly incremental, Daily full.
  - Models: Versioned in S3/Artifactory.

## 7. Checklist for Go-Live

- [ ] Security Audit passed (0 Critical/High).
- [ ] License Check passed.
- [ ] SBOM generated and stored.
- [ ] Secrets rotated and injected securely.
- [ ] Monitoring alerts configured.
- [ ] Disaster Recovery drill performed.
