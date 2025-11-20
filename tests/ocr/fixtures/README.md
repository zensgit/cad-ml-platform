# OCR Test Fixtures

Test fixtures for OCR fallback parsing validation.

## Directory Structure

```
fixtures/
└── deepseek_mock_output/
    ├── valid_json.txt          # Perfect JSON output (Level 1)
    ├── markdown_fence.txt      # JSON in markdown fence (Level 2)
    ├── malformed_json.txt      # Syntax errors, triggers fallback
    ├── text_only.txt           # Plain text requiring regex (Level 3)
    └── bom_mixed.txt           # UTF-8 BOM with mixed content
```

## Fixture Usage

### valid_json.txt
- **Purpose**: Test Level 1 (JSON_STRICT) parsing
- **Expected Result**: Direct JSON parsing success
- **Fallback Level**: `JSON_STRICT`

### markdown_fence.txt
- **Purpose**: Test Level 2 (MARKDOWN_FENCE) parsing
- **Expected Result**: Extract JSON from ```json blocks
- **Fallback Level**: `MARKDOWN_FENCE`

### malformed_json.txt
- **Purpose**: Test JSON syntax error recovery
- **Expected Result**: Falls through to Level 2 or 3
- **Fallback Level**: `MARKDOWN_FENCE` or `TEXT_REGEX`
- **Note**: Contains trailing commas (common LLM error)

### text_only.txt
- **Purpose**: Test Level 3 (TEXT_REGEX) parsing
- **Expected Result**: Extract dimensions/symbols via regex patterns
- **Fallback Level**: `TEXT_REGEX`
- **Coverage**:
  - Chinese text extraction
  - Bidirectional tolerance (±0.02)
  - Thread pitch (M10×1.5)
  - Title block fields (图号, 材料, 名称)

### bom_mixed.txt
- **Purpose**: Test BOM handling + mixed content
- **Expected Result**: Successfully parse despite BOM and extra text
- **Fallback Level**: `MARKDOWN_FENCE`
- **Note**: Simulates real DeepSeek output with UTF-8 BOM prefix

## Integration

Load fixtures in tests:

```python
import pytest
from pathlib import Path

@pytest.fixture
def fixture_dir():
    return Path(__file__).parent / "fixtures" / "deepseek_mock_output"

def test_with_fixture(fixture_dir):
    output = (fixture_dir / "valid_json.txt").read_text()
    result = parser.parse(output)
    assert result.success
```

## Updating Fixtures

When updating DeepSeek prompt versions or output format:

1. Add new fixture file with version suffix (e.g., `valid_json_v2.txt`)
2. Update tests to use versioned fixtures
3. Keep old fixtures for regression testing
4. Document changes in CHANGELOG.md
