"""
Test L3 Fusion Flow.

Verifies that the FusionClassifier correctly prioritizes 3D signals 
over 2D/Text signals when confident 3D features are present.
"""

import unittest
from src.core.knowledge.fusion import get_fusion_classifier
from src.ml.vision_3d import get_3d_encoder

class TestL3Fusion(unittest.TestCase):
    
    def setUp(self):
        self.fusion = get_fusion_classifier()
        self.encoder = get_3d_encoder()

    def test_mock_embedding_determinism(self):
        """Test that the mock encoder is deterministic."""
        feat1 = {"faces": 100, "volume": 50.0}
        feat2 = {"faces": 100, "volume": 50.0}
        
        vec1 = self.encoder.encode(feat1)
        vec2 = self.encoder.encode(feat2)
        
        self.assertEqual(vec1, vec2)
        self.assertTrue(len(vec1) > 0)

    def test_3d_signal_override(self):
        """
        Scenario: 
        - Text says 'Plate' (incorrectly labeled).
        - 3D Topology is clearly a 'Shaft' (Cylindrical).
        - Expectation: Fusion should favor 'Shaft' or at least give it high confidence.
        """
        
        # 1. 2D Signals (Ambiguous or Misleading)
        text_signals = "Steel Plate"
        features_2d = {
            "geometric_features": {"aspect_ratio": 2.0},
            "entity_counts": {"LINE": 10, "CIRCLE": 2} # Could be anything
        }
        
        # 2. 3D Signals (Strong 'Shaft' indicator)
        # Mocking features that _analyze_3d_signals looks for
        features_3d = {
            "valid_3d": True,
            "faces": 20,
            "surface_types": {
                "cylinder": 15, # 75% Cylindrical -> Shaft
                "plane": 5
            }
        }
        
        # 3. Classify
        result = self.fusion.classify(text_signals, features_2d, features_3d)
        
        # 4. Assert
        print(f"Fusion Result: {result}")
        
        # The logic in fusion.py sets shaft=0.6 if cyl_ratio > 0.4
        # Even if text says "Plate", the 3D signal should push Shaft up.
        self.assertEqual(result["type"], "shaft")
        self.assertGreater(result["confidence"], 0.4)
        self.assertEqual(result["fusion_breakdown"]["source"], "l3_fusion")
        self.assertGreater(result["fusion_breakdown"]["3d_score"], 0.5)

    def test_fusion_no_3d(self):
        """Test fallback when 3D is invalid."""
        result = self.fusion.classify(
            "Bearing", 
            {"geometric_features": {}, "entity_counts": {}}, 
            {"valid_3d": False}
        )
        # Should rely purely on text/2d
        # Assuming KM has a rule for 'Bearing' -> 'bearing'
        # Note: Depending on KM state, this might vary, but we check structure
        self.assertIn("type", result)
        self.assertEqual(result["fusion_breakdown"]["3d_score"], 0)

if __name__ == "__main__":
    unittest.main()
