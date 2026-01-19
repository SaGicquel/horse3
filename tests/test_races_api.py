import pytest
from datetime import date
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_healthy(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestRacesToday:
    def test_races_today_returns_200(self):
        response = client.get("/races/today")
        assert response.status_code == 200
        data = response.json()
        assert "date" in data
        assert "nombre_courses" in data
        assert "courses" in data
        assert isinstance(data["courses"], list)

    def test_races_today_has_correct_date(self):
        response = client.get("/races/today")
        data = response.json()
        assert data["date"] == str(date.today())


class TestRacesByDate:
    def test_races_by_date_returns_200(self):
        response = client.get("/races/date/2025-01-15")
        assert response.status_code == 200
        data = response.json()
        assert data["date"] == "2025-01-15"

    def test_races_by_date_invalid_format(self):
        response = client.get("/races/date/invalid-date")
        assert response.status_code == 422


class TestRaceDetail:
    def test_race_not_found_returns_404(self):
        response = client.get("/races/NONEXISTENT_ID_12345")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestRacesList:
    def test_list_races_returns_paginated(self):
        response = client.get("/races/")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "page" in data
        assert "per_page" in data
        assert "pages" in data
        assert "items" in data

    def test_list_races_pagination_params(self):
        response = client.get("/races/?page=1&per_page=5")
        assert response.status_code == 200
        data = response.json()
        assert data["per_page"] == 5
        assert len(data["items"]) <= 5

    def test_list_races_filter_discipline(self):
        response = client.get("/races/?discipline=TROT")
        assert response.status_code == 200

    def test_list_races_filter_date_range(self):
        response = client.get("/races/?date_from=2025-01-01&date_to=2025-01-15")
        assert response.status_code == 200
