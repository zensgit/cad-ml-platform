# L4 Features: Usage Guide

This guide explains how to use the advanced L4 capabilities of the CAD ML Platform: **DFM Analysis**, **Process Recommendation**, and **Cost Estimation**.

## 1. Enabling L4 Analysis

To trigger L4 analysis, you must upload a 3D file (`STEP`, `IGES`) and enable the corresponding flags in the `options` JSON.

```bash
POST /api/v1/analyze
File: gear_box.step
Options:
{
  "extract_features": true,
  "classify_parts": true,
  "quality_check": true,       // Triggers DFM Analysis
  "process_recommendation": true, // Triggers Process Selection
  "estimate_cost": true        // Triggers Cost Estimation
}
```

## 2. Configuration (`manufacturing_data.yaml`)

The system logic is driven by `config/manufacturing_data.yaml`. You can hot-reload parameters by updating this file.

### Material Pricing
Adjust material costs (USD/kg) and densities.
```yaml
materials:
  steel:
    price_per_kg_usd: 2.5
    density_kg_per_m3: 7850
```

### Machine Rates
Adjust hourly rates for different processes.
```yaml
machine_hourly_rates:
  cnc_milling: 60.0  # 3-axis
  5_axis: 120.0      # Complex milling
```

### DFM Thresholds
Control when warnings are triggered.
```yaml
dfm_thresholds:
  min_wall_thickness_mm: 0.8  # Walls thinner than this trigger THIN_WALL warning
  max_stock_removal_ratio: 0.85 # Removal > 85% triggers HIGH_WASTE warning
```

## 3. Interpreting Results

### DFM Report (`quality`)
```json
"quality": {
  "mode": "L4_DFM",
  "score": 80.0,
  "issues": [
    {
      "code": "THIN_WALL",
      "severity": "high",
      "message": "Potential thin walls detected (~0.50mm)..."
    }
  ]
}
```

### Process Recommendation (`process`)
```json
"process": {
  "primary_recommendation": {
    "process": "cnc_milling",
    "method": "5_axis",
    "confidence": 0.9,
    "reason": "Prismatic geometry suitable for milling."
  }
}
```

### Cost Estimation (`cost_estimation`)
```json
"cost_estimation": {
  "total_unit_cost": 45.20,
  "breakdown": {
    "material_cost": 9.37,
    "machining_cost": 32.50,
    "setup_amortized": 3.33
  }
}
```

## 4. Feature Vector Order (v1-v4)

When features are flattened for similarity storage, the canonical order is:

1) Base geometric: entity_count, bbox_width, bbox_height, bbox_depth, bbox_volume_estimate
2) Semantic: layer_count, complexity_high_flag
3) Geometric extensions: v2 then v3 then v4

This matches `FeatureExtractor.flatten()` and `FeatureExtractor.rehydrate()`.

### Vector Layout Metadata

Vectors carry a `vector_layout` tag in metadata to disambiguate storage order:

- `base_sem_ext_v1`: canonical 2D layout (base + semantic + extensions).
- `base_sem_ext_v1+l3`: canonical 2D layout with an appended L3 embedding tail.
  - The tail size is stored in `l3_3d_dim`.
- `geom_all_sem_v1`: legacy layout (geometric-all + semantic) that is reordered on migration.

When L3 embedding is present, migration updates only the 2D portion and preserves the L3 tail.
