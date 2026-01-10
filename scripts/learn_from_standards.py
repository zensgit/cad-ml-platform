"""
Learned Rules Loader.

Script to scan the standards library (DXF), extract features,
and generate new Knowledge Rules based on the examples.
"""

import os
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.feature_extractor import FeatureExtractor
from src.models.cad_document import CadDocument, CadEntity, BoundingBox
from src.core.knowledge.dynamic.models import KnowledgeEntry, KnowledgeCategory
from src.core.knowledge.dynamic.manager import get_knowledge_manager
from src.adapters.factory import DxfAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StandardLearner")

STANDARDS_DIR = "data/standards_dxf"

async def learn_from_standards():
    if not os.path.exists(STANDARDS_DIR):
        logger.error(f"Standards dir {STANDARDS_DIR} not found.")
        return

    logger.info("Starting active learning from standards...")
    km = get_knowledge_manager()
    extractor = FeatureExtractor()
    adapter = DxfAdapter() # Correct adapter class
    
    files = [f for f in os.listdir(STANDARDS_DIR) if f.lower().endswith(".dxf")]
    
    new_rules_count = 0
    
    for f in files:
        path = os.path.join(STANDARDS_DIR, f)
        try:
            # 1. Parse File
            with open(path, 'rb') as file_obj:
                content = file_obj.read()
                
            # Adapter returns doc (async)
            # Since adapter.parse is async, we await it
            # Mocking adapter behavior if real one is complex for this script context
            # Real DXFAdapter uses ezdxf to populate CadDocument
            
            # Let's try to use the real adapter if possible, otherwise mock parsing for the script
            try:
                doc = await adapter.parse(content, f)
            except Exception:
                # Simple fallback parsing for the script (ezdxf directly)
                import ezdxf
                dxf = ezdxf.readfile(path)
                msp = dxf.modelspace()
                entities = []
                for e in msp:
                    entities.append(CadEntity(kind=e.dxftype()))
                # Bounds
                # ezdxf bounds calculation... placeholder
                doc = CadDocument(file_name=f, format="dxf", entities=entities)
                # doc.entity_count() etc will work
            
            # 2. Extract Features
            feats = await extractor.extract(doc)
            geo = feats["geometric"]
            counts = feats.get("entity_counts", {})
            
            # 3. Generate Rule
            # Heuristic: Name "Bolt_M6..." -> Part Type "bolt"
            part_type = "unknown"
            if "bolt" in f.lower(): part_type = "bolt"
            elif "washer" in f.lower(): part_type = "washer"
            elif "flange" in f.lower(): part_type = "flange"
            
            if part_type == "unknown":
                continue
                
            # Create a rule based on entity counts
            # e.g. If input has similar circle/line ratio
            
            # Simplified Logic:
            # Create a keyword rule for the filename pattern (Strongest signal here)
            # In a real ML system, we'd add the vector to a classifier.
            # Here we demonstrate updating the KnowledgeBase.
            
            rule_id = f"learned_{part_type}_{f.split('.')[0]}"
            
            # Check if exists
            if km.get_rule(rule_id):
                continue
                
            rule = KnowledgeEntry(
                id=rule_id,
                category=KnowledgeCategory.GEOMETRY, # Using Geometry category for shape-based rules
                name=f"Learned {part_type} from {f}",
                keywords=[f.split('.')[0], part_type], # Keywords matching filename parts
                ocr_patterns=[],
                part_hints={part_type: 0.95}, # High confidence for exact match
                enabled=True,
                source="active_learning"
            )
            
            km.add_rule(rule)
            new_rules_count += 1
            logger.info(f"Learned new rule: {rule_id}")
            
        except Exception as e:
            logger.error(f"Failed to learn from {f}: {e}")
            
    logger.info(f"Learning complete. Added {new_rules_count} new rules.")

if __name__ == "__main__":
    asyncio.run(learn_from_standards())
