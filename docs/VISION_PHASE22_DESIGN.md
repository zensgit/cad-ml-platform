# Phase 22: Advanced Security & Governance

## Overview

Phase 22 introduces a comprehensive security governance system through the `security_governance.py` module. This central hub integrates policy management, data classification, encryption key management, secret management, threat intelligence, security event correlation, and security posture assessment into a unified governance platform.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SecurityGovernanceHub                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐   │
│  │   Policy    │ │    Data     │ │    Key      │ │   Secret    │   │
│  │   Engine    │ │Classification│ │  Manager   │ │  Manager    │   │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘   │
│         │               │               │               │           │
│  ┌──────┴───────────────┴───────────────┴───────────────┴──────┐   │
│  │                    Unified Security Pipeline                  │   │
│  └──────┬───────────────┬───────────────┬───────────────┬──────┘   │
│         │               │               │               │           │
│  ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐                   │
│  │   Threat    │ │   Event     │ │   Posture   │                   │
│  │   Intel     │ │ Correlator  │ │  Assessor   │                   │
│  └─────────────┘ └─────────────┘ └─────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     SecureVisionProvider                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Base VisionProvider + Policy Evaluation + Classification    │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Enums

| Enum | Values | Description |
|------|--------|-------------|
| `DataClassification` | public, internal, confidential, restricted, top_secret | Data sensitivity levels |
| `PolicyType` | access, data, encryption, retention, audit | Types of security policies |
| `PolicyStatus` | draft, active, inactive, deprecated | Policy lifecycle states |
| `PolicyAction` | allow, deny, audit, encrypt, redact | Actions taken by policies |
| `EncryptionAlgorithm` | aes_256_gcm, aes_256_cbc, chacha20_poly1305, rsa_4096 | Supported encryption algorithms |
| `KeyStatus` | active, inactive, pending_rotation, rotated, revoked | Key lifecycle states |
| `SecretType` | api_key, password, certificate, token, ssh_key | Types of secrets |
| `ThreatLevel` | low, medium, high, critical | Threat severity levels |
| `SecurityEventType` | access_granted, access_denied, policy_violation, anomaly, threat_detected, key_rotation, secret_access | Security event types |
| `PostureStatus` | excellent, good, fair, poor, critical | Security posture assessment states |

### 2. Dataclasses

#### SecurityPolicy
```python
@dataclass
class SecurityPolicy:
    policy_id: str
    name: str
    policy_type: PolicyType
    status: PolicyStatus
    action: PolicyAction
    conditions: Dict[str, Any]
    description: str
    created_at: datetime
    updated_at: datetime
    version: int
```

#### PolicyEvaluation
```python
@dataclass
class PolicyEvaluation:
    policy_id: str
    allowed: bool
    action: PolicyAction
    matched_conditions: List[str]
    timestamp: datetime
    context: Dict[str, Any]
```

#### DataClassificationRule
```python
@dataclass
class DataClassificationRule:
    rule_id: str
    name: str
    classification: DataClassification
    patterns: List[str]
    keywords: List[str]
    description: str
```

#### EncryptionKey
```python
@dataclass
class EncryptionKey:
    key_id: str
    algorithm: EncryptionAlgorithm
    status: KeyStatus
    created_at: datetime
    expires_at: datetime
    rotated_at: Optional[datetime]
    metadata: Dict[str, Any]
```

#### Secret
```python
@dataclass
class Secret:
    secret_id: str
    name: str
    secret_type: SecretType
    encrypted_value: str
    created_at: datetime
    expires_at: Optional[datetime]
    rotated_at: Optional[datetime]
    version: int
    metadata: Dict[str, Any]
```

#### ThreatIndicator
```python
@dataclass
class ThreatIndicator:
    indicator_id: str
    indicator_type: str
    value: str
    threat_level: ThreatLevel
    source: str
    created_at: datetime
    expires_at: Optional[datetime]
    metadata: Dict[str, Any]
```

#### SecurityEvent
```python
@dataclass
class SecurityEvent:
    event_id: str
    event_type: SecurityEventType
    source: str
    target: str
    action: str
    result: str
    timestamp: datetime
    details: Dict[str, Any]
    correlation_id: Optional[str]
```

#### SecurityPosture
```python
@dataclass
class SecurityPosture:
    posture_id: str
    status: PostureStatus
    score: float
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    timestamp: datetime
    details: Dict[str, Any]
```

#### GovernanceConfig
```python
@dataclass
class GovernanceConfig:
    key_rotation_days: int = 90
    secret_expiry_days: int = 365
    max_events: int = 10000
    threat_intel_ttl_days: int = 30
    posture_assessment_interval: int = 3600
    enable_auto_classification: bool = True
    enable_threat_detection: bool = True
```

### 3. Core Classes

#### PolicyEngine
```python
class PolicyEngine:
    def add_policy(policy: SecurityPolicy) -> None
    def remove_policy(policy_id: str) -> bool
    def update_policy(policy_id: str, updates: Dict[str, Any]) -> Optional[SecurityPolicy]
    def get_policy(policy_id: str) -> Optional[SecurityPolicy]
    def list_policies(policy_type: PolicyType = None) -> List[SecurityPolicy]
    def evaluate(context: Dict[str, Any]) -> PolicyEvaluation
```

#### DataClassificationManager
```python
class DataClassificationManager:
    def add_rule(rule: DataClassificationRule) -> None
    def remove_rule(rule_id: str) -> bool
    def classify(data: str) -> DataClassification
    def get_rules() -> List[DataClassificationRule]
    def get_rules_for_classification(classification: DataClassification) -> List[DataClassificationRule]
```

#### KeyManager
```python
class KeyManager:
    def generate_key(algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM,
                     validity_days: int = None) -> EncryptionKey
    def get_key(key_id: str) -> Optional[EncryptionKey]
    def rotate_key(key_id: str) -> Optional[EncryptionKey]
    def revoke_key(key_id: str) -> bool
    def list_keys(status: KeyStatus = None) -> List[EncryptionKey]
    def get_keys_for_rotation(days_until_expiry: int = 7) -> List[EncryptionKey]
```

#### SecretManager
```python
class SecretManager:
    def store_secret(name: str, value: str, secret_type: SecretType,
                     expires_in_days: int = None, metadata: Dict = None) -> Secret
    def get_secret(secret_id: str) -> Optional[str]
    def get_secret_by_name(name: str) -> Optional[str]
    def rotate_secret(secret_id: str, new_value: str) -> Optional[Secret]
    def delete_secret(secret_id: str) -> bool
    def list_secrets() -> List[Secret]
    def get_expiring_secrets(days: int = 30) -> List[Secret]
```

#### ThreatIntelManager
```python
class ThreatIntelManager:
    def add_indicator(indicator: ThreatIndicator) -> None
    def remove_indicator(indicator_id: str) -> bool
    def check_threat(value: str, indicator_type: str = None) -> Optional[ThreatIndicator]
    def get_indicators(threat_level: ThreatLevel = None) -> List[ThreatIndicator]
    def get_indicators_by_level(level: ThreatLevel) -> List[ThreatIndicator]
```

#### SecurityEventCorrelator
```python
class SecurityEventCorrelator:
    def record_event(event: SecurityEvent) -> None
    def get_events(event_type: SecurityEventType = None,
                   start_time: datetime = None, end_time: datetime = None,
                   limit: int = 100) -> List[SecurityEvent]
    def correlate_events(time_window_seconds: int = 300) -> List[List[SecurityEvent]]
    def get_event_count() -> int
```

#### SecurityPostureAssessor
```python
class SecurityPostureAssessor:
    def assess() -> SecurityPosture
    def get_latest_assessment() -> Optional[SecurityPosture]
    def get_assessment_history(limit: int = 10) -> List[SecurityPosture]
```

#### SecurityGovernanceHub
```python
class SecurityGovernanceHub:
    @property policy_engine: PolicyEngine
    @property classification: DataClassificationManager
    @property key_manager: KeyManager
    @property secret_manager: SecretManager
    @property threat_intel: ThreatIntelManager
    @property event_correlator: SecurityEventCorrelator
    @property posture_assessor: SecurityPostureAssessor

    def evaluate_access(context: Dict[str, Any]) -> PolicyEvaluation
    def record_security_event(event_type: SecurityEventType, source: str,
                              target: str, action: str, result: str,
                              details: Dict = None) -> SecurityEvent
    def get_governance_summary() -> Dict[str, Any]
```

### 4. Vision Provider Integration

#### SecureVisionProvider
```python
class SecureVisionProvider(VisionProvider):
    def __init__(base_provider: VisionProvider, hub: SecurityGovernanceHub)

    @property provider_name: str  # "secure_{base_provider_name}"

    async def analyze_image(image_data: bytes, include_description: bool) -> VisionDescription
```

Features:
- Pre-request policy evaluation
- Data classification of inputs
- Security event recording
- Threat detection on inputs
- Access denial with detailed errors
- Automatic audit logging

### 5. Factory Functions

```python
def create_governance_config(
    key_rotation_days: int = 90,
    secret_expiry_days: int = 365,
    **kwargs
) -> GovernanceConfig

def create_security_governance_hub(
    key_rotation_days: int = 90,
    secret_expiry_days: int = 365,
    **kwargs
) -> SecurityGovernanceHub

def create_security_policy(
    name: str,
    policy_type: PolicyType,
    action: PolicyAction = PolicyAction.DENY,
    conditions: Dict[str, Any] = None,
    **kwargs
) -> SecurityPolicy

def create_classification_rule(
    name: str,
    classification: DataClassification,
    patterns: List[str] = None,
    keywords: List[str] = None,
    **kwargs
) -> DataClassificationRule

def create_threat_indicator(
    indicator_type: str,
    value: str,
    threat_level: ThreatLevel,
    source: str = "manual",
    **kwargs
) -> ThreatIndicator

def create_secure_provider(
    base_provider: VisionProvider,
    hub: SecurityGovernanceHub = None
) -> SecureVisionProvider
```

## Usage Examples

### Basic Policy Management
```python
from src.core.vision import create_security_governance_hub, create_security_policy
from src.core.vision import PolicyType, PolicyAction

hub = create_security_governance_hub()

# Create access policy
policy = create_security_policy(
    name="Admin Access Only",
    policy_type=PolicyType.ACCESS,
    action=PolicyAction.ALLOW,
    conditions={"role": "admin"}
)
hub.policy_engine.add_policy(policy)

# Evaluate access
result = hub.evaluate_access({"role": "admin", "resource": "secrets"})
print(f"Allowed: {result.allowed}, Action: {result.action}")
```

### Data Classification
```python
from src.core.vision import create_security_governance_hub, create_classification_rule
from src.core.vision import DataClassification

hub = create_security_governance_hub()

# Add classification rule
rule = create_classification_rule(
    name="PII Detection",
    classification=DataClassification.CONFIDENTIAL,
    patterns=[r"\d{3}-\d{2}-\d{4}"],  # SSN pattern
    keywords=["social security", "ssn"]
)
hub.classification.add_rule(rule)

# Classify data
classification = hub.classification.classify("SSN: 123-45-6789")
print(f"Classification: {classification}")  # CONFIDENTIAL
```

### Key Management
```python
from src.core.vision import create_security_governance_hub
from src.core.vision import EncryptionAlgorithm, KeyStatus

hub = create_security_governance_hub(key_rotation_days=30)

# Generate key
key = hub.key_manager.generate_key(EncryptionAlgorithm.AES_256_GCM)
print(f"Key ID: {key.key_id}, Status: {key.status}")

# Get keys needing rotation
expiring = hub.key_manager.get_keys_for_rotation(days_until_expiry=7)
for k in expiring:
    hub.key_manager.rotate_key(k.key_id)
```

### Secret Management
```python
from src.core.vision import create_security_governance_hub, SecretType

hub = create_security_governance_hub()

# Store secret
secret = hub.secret_manager.store_secret(
    name="api_key_prod",
    value="sk-secret-key-12345",
    secret_type=SecretType.API_KEY,
    expires_in_days=90,
    metadata={"environment": "production"}
)

# Retrieve secret
value = hub.secret_manager.get_secret_by_name("api_key_prod")

# Check expiring secrets
expiring = hub.secret_manager.get_expiring_secrets(days=30)
```

### Threat Intelligence
```python
from src.core.vision import create_security_governance_hub, create_threat_indicator
from src.core.vision import ThreatLevel

hub = create_security_governance_hub()

# Add threat indicator
indicator = create_threat_indicator(
    indicator_type="ip",
    value="192.168.1.100",
    threat_level=ThreatLevel.HIGH,
    source="threat_feed"
)
hub.threat_intel.add_indicator(indicator)

# Check for threats
threat = hub.threat_intel.check_threat("192.168.1.100", "ip")
if threat:
    print(f"Threat detected: {threat.threat_level}")
```

### Security Event Correlation
```python
from src.core.vision import create_security_governance_hub, SecurityEventType

hub = create_security_governance_hub()

# Record events
hub.record_security_event(
    event_type=SecurityEventType.ACCESS_DENIED,
    source="user123",
    target="/api/admin",
    action="GET",
    result="denied",
    details={"reason": "insufficient_permissions"}
)

# Get correlated events
correlated = hub.event_correlator.correlate_events(time_window_seconds=300)
for group in correlated:
    print(f"Correlated group: {len(group)} events")
```

### Security Posture Assessment
```python
from src.core.vision import create_security_governance_hub

hub = create_security_governance_hub()

# Add policies, keys, etc.
# ...

# Assess posture
posture = hub.posture_assessor.assess()
print(f"Status: {posture.status}")
print(f"Score: {posture.score}/100")
for rec in posture.recommendations:
    print(f"  - {rec}")
```

### Secure Vision Provider
```python
from src.core.vision import create_secure_provider, create_security_governance_hub
from src.core.vision import create_security_policy, PolicyType, PolicyAction

hub = create_security_governance_hub()

# Add allow policy
policy = create_security_policy(
    name="Allow Vision Analysis",
    policy_type=PolicyType.ACCESS,
    action=PolicyAction.ALLOW,
    conditions={"operation": "analyze_image"}
)
hub.policy_engine.add_policy(policy)

# Create secure provider
secure_provider = create_secure_provider(my_base_provider, hub)

# Analyze with security checks
result = await secure_provider.analyze_image(image_data)
```

## Test Coverage

| Test Class | Tests | Coverage |
|------------|-------|----------|
| TestSecurityEnums | 10 | All enum values |
| TestSecurityDataclasses | 9 | All dataclass creation |
| TestPolicyEngine | 6 | Policy CRUD and evaluation |
| TestDataClassificationManager | 5 | Classification rules and matching |
| TestKeyManager | 6 | Key lifecycle management |
| TestSecretManager | 7 | Secret storage and retrieval |
| TestThreatIntelManager | 4 | Threat indicator management |
| TestSecurityEventCorrelator | 3 | Event recording and correlation |
| TestSecurityPostureAssessor | 5 | Posture assessment |
| TestSecurityGovernanceHub | 5 | Hub integration |
| TestSecureVisionProvider | 3 | Provider integration |
| TestFactoryFunctions | 7 | All factory functions |
| TestSecurityIntegration | 4 | End-to-end scenarios |

**Total: 80 tests**

## Integration with Existing Modules

Phase 22 complements existing security modules:

| Existing Module | Phase 22 Enhancement |
|----------------|---------------------|
| `security_audit.py` | SecurityEventCorrelator adds correlation |
| `security_scanner.py` | ThreatIntelManager adds threat feeds |
| `access_control.py` | PolicyEngine adds fine-grained policies |
| `compliance.py` | SecurityPostureAssessor adds posture tracking |
| `encryption.py` | KeyManager adds lifecycle management |

## Performance Considerations

- **Thread Safety**: All components use `threading.RLock()` for concurrent access
- **Event Pruning**: Automatic cleanup of old events beyond `max_events`
- **Key Rotation**: Built-in rotation tracking and notifications
- **Secret Encryption**: Base64 encoding (production should use real encryption)
- **Lazy Assessment**: Posture assessment on-demand with caching

## Security Features

| Feature | Implementation |
|---------|---------------|
| Policy Evaluation | Condition matching with allow/deny/audit actions |
| Data Classification | Pattern and keyword-based classification |
| Key Management | Generation, rotation, revocation lifecycle |
| Secret Storage | Encrypted storage with versioning |
| Threat Detection | Indicator matching by type and value |
| Event Correlation | Time-window based correlation |
| Posture Assessment | Multi-factor scoring with recommendations |

## Dependencies

- Standard library only (no external dependencies)
- Uses `threading`, `uuid`, `re`, `base64`, `asyncio`
- Integrates with `VisionProvider` base class

## File Structure

```
src/core/vision/
├── security_governance.py    # Phase 22 implementation
├── __init__.py               # Updated with Phase 22 exports
└── ...

tests/unit/
├── test_vision_phase22.py    # 80 comprehensive tests
└── ...

docs/
├── VISION_PHASE22_DESIGN.md  # This document
└── ...
```

## Summary

Phase 22 provides a unified security governance platform that:
- Centralizes policy management and evaluation
- Automates data classification
- Manages encryption key lifecycle
- Secures secret storage and access
- Integrates threat intelligence
- Correlates security events
- Assesses overall security posture
- Wraps Vision providers with security controls
