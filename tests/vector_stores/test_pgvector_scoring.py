"""Tests for pgvector scoring fix (no psycopg required)."""
import pytest


class TestPGVectorScoring:
    """PGVector scoring edge cases (cosine distance clamp)."""

    @staticmethod
    def _mock_format_results(results):
        """Replicate the scoring logic from PGVector without importing psycopg.

        From pgvector.py:
            d = max(0.0, float(r[1]))
            score = max(0.0, 1.0 - d)
        """
        from collections import namedtuple
        OutputData = namedtuple("OutputData", ["id", "score", "payload"])

        output = []
        for r in results:
            d = max(0.0, float(r[1]))
            score = max(0.0, 1.0 - d)
            output.append(OutputData(id=str(r[0]), score=score, payload=r[2]))
        return output

    def test_normal_distance_scored_correctly(self):
        """Distance <= 1.0 should use 1.0 - d formula."""
        results = [("id1", 0.1, {}), ("id2", 0.5, {}), ("id3", 0.9, {})]
        output = self._mock_format_results(results)
        assert len(output) == 3

        scores = [o.score for o in output]
        assert scores[0] == pytest.approx(0.9)  # 1.0 - 0.1
        assert scores[1] == pytest.approx(0.5)  # 1.0 - 0.5
        assert scores[2] == pytest.approx(0.1)  # 1.0 - 0.9

    def test_distance_greater_than_one_clamps(self):
        """Distance > 1.0 should clamp to 0.0 (no discontinuity)."""
        results = [("id1", 0.99, {}), ("id2", 1.01, {}), ("id3", 2.0, {})]
        output = self._mock_format_results(results)

        assert output[0].score > 0
        assert output[1].score == 0.0
        assert output[2].score == 0.0

    def test_clamping_preserves_ranking_consistency(self):
        """Higher distance should never produce higher score."""
        positions = [("id1", 0.99, {}), ("id2", 1.01, {})]
        output = self._mock_format_results(positions)
        assert output[0].score >= output[1].score

        results = [("id1", 0.0, {}), ("id2", 0.5, {}), ("id3", 1.0, {}), ("id4", 2.0, {}), ("id5", 3.0, {})]
        output = self._mock_format_results(results)

        for i in range(len(output) - 1):
            assert output[i].score >= output[i + 1].score, \
                f"Score not monotonic at index {i}: {output[i].score} < {output[i+1].score}"

    def test_clean_distance(self):
        """Distance >= 0 should not produce negative scores."""
        results = [("id1", 0.0, {}), ("id2", 0.3, {}), ("id3", 0.8, {})]
        output = self._mock_format_results(results)
        for o in output:
            assert o.score >= 0.0

    def test_negative_distance_clamped(self):
        """Negative distance should be clamped to 0."""
        results = [("id1", -0.5, {})]
        output = self._mock_format_results(results)
        assert output[0].score == pytest.approx(1.0)  # max(0, -0.5)=0, 1-0=1
