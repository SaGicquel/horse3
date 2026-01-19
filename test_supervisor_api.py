import unittest
from unittest.mock import MagicMock, patch
import sys
import json
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from api_prediction import app, CourseRequest, ChevalFeatures
from fastapi.testclient import TestClient

client = TestClient(app)


class TestSupervisorAPI(unittest.TestCase):
    @patch("api_prediction.supervisor")
    @patch("api_prediction.model_manager")
    def test_analyze_endpoint(self, mock_model_manager, mock_supervisor):
        # Setup mocks
        mock_model_manager.predict.return_value = (
            [
                {"numero_partant": 1, "probabilite_victoire": 0.3, "rang_prediction": 1},
                {"numero_partant": 2, "probabilite_victoire": 0.2, "rang_prediction": 2},
            ],
            10.0,
            "v1",
        )

        # Mock supervisor result
        from ai_supervisor import SupervisorResult

        mock_supervisor.analyze.return_value = SupervisorResult(
            course_id="TEST_COURSE",
            timestamp="2026-01-01T12:00:00",
            analysis="Analysis text",
            anomalies=[{"type": "TEST", "severity": "HIGH", "detail": "Test detail"}],
            recommendations=["Rec 1", "Rec 2"],
            confidence_score=0.85,
            provider="mock",
        )

        # Request data
        payload = {
            "course_id": "TEST_COURSE",
            "date_course": "2026-01-01",
            "hippodrome": "VINCENNES",
            "distance": 2700,
            "type_piste": "Trot",
            "partants": [
                {
                    "cheval_id": "C1",
                    "numero_partant": 1,
                    "forme_5c": 0.8,
                    "forme_10c": 0.7,
                    "nb_courses_12m": 10,
                    "nb_victoires_12m": 2,
                    "cote_sp": 2.5,
                },
                {
                    "cheval_id": "C2",
                    "numero_partant": 2,
                    "forme_5c": 0.5,
                    "forme_10c": 0.5,
                    "nb_courses_12m": 12,
                    "nb_victoires_12m": 1,
                    "cote_sp": 5.0,
                },
            ],
        }

        response = client.post("/analyze", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["course_id"], "TEST_COURSE")
        self.assertEqual(data["provider"], "mock")
        self.assertEqual(len(data["anomalies"]), 1)
        self.assertEqual(data["confidence_score"], 0.85)


if __name__ == "__main__":
    unittest.main()
