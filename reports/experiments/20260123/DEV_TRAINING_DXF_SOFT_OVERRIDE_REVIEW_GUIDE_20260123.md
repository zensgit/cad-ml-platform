# DEV_TRAINING_DXF_SOFT_OVERRIDE_REVIEW_GUIDE_20260123

## Purpose
Provide a lightweight, consistent checklist for manually reviewing the 12 soft-override candidates.

## How to Review (per file)
1. Open the DXF in your preferred viewer.
2. Look for dominant geometry cues:
   - 传动件: shafts, gears, couplings, rotating assemblies.
   - 罐体: cylindrical tanks, vessels, pressure shells.
   - 设备: packaged systems, trailers, machinery frames.
3. Decide if Graph2D label matches the drawing's primary function.
4. If not, assign the correct label and note why (e.g., "overall assembly", "sub-part", "multi-function").

## Common Pitfalls
- Multi-part assemblies can resemble 设备 even when the dominant sub-part is 传动件.
- Cylinder-heavy assemblies may be tagged as 罐体 even if they are frames or supports.
- Names like "组件" or "法兰" do not always reflect the dominant geometry.

## Recommended Fields to Fill
- agree_with_graph2d: yes/no
- correct_label: the label you believe is most accurate
- notes: short reason (1–2 sentences)

## Output Location
- Review template: `reports/experiments/20260123/dxf_batch_analysis_training_dxf_oda_graph2d_node19_normalized_cleaned_augmented_soft_override_suggestion/soft_override_eligible_review_template.csv`
