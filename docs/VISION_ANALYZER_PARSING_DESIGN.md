# Vision Analyzer Parsing Design

## Overview
The vision analyzer text parsing is a lightweight extraction layer that normalizes model
responses into structured fields used by downstream CAD workflows. It accepts:
- JSON payloads (direct or wrapped in fenced code blocks).
- Mixed plain-text responses from vision providers.

## Goals
- Extract core fields (objects, text, CAD elements, dimensions) with minimal assumptions.
- Tolerate imperfect provider output (partial JSON, mixed prose).
- Keep logic dependency-free (stdlib only) and safe for untrusted input.

## Parsing Strategy
1. JSON-first: attempt to parse the entire response as JSON.
2. Fenced JSON: scan for ```json blocks and parse the first valid JSON block.
3. Embedded snippet: detect the first balanced JSON object/array in the text.
4. Plain text fallback: extract objects from bullet/numbered lists and recover
   dimensions/tolerances via regex.

## Output Shape (Current)
- objects: list of dicts (normalized from strings/dicts/lists)
- text: string (OCR or extracted text)
- drawings/cad_elements: dict or {"items": [..]} fallback
- dimensions: dict with optional fields:
  - values: list of {value, unit, type?}
  - tolerances: list of {type, value, unit?} or {type, plus, minus, unit?}

## Supported Dimension Patterns
- Absolute values: `10 mm`, `0.5 in`
- Diameter: `Ø10`, `∅6.35`
- Symmetric tolerance: `±0.05`, `+/- 0.1`, `±0.05mm`
- Asymmetric tolerance: `+0.1/-0.05`, `(+0.1/-0.02)`

## Known Limitations
- No unit conversion or inference (units are preserved as-is).
- Duplicate values may appear if the same number is represented in multiple forms.
- No geometry validation beyond basic token parsing.

## Extension Points
- Add unit normalization (e.g., "inch" -> "in").
- Deduplicate values and tolerances by proximity or context.
- Add CAD symbol parsing (GD&T, surface finish) as separate extractors.
