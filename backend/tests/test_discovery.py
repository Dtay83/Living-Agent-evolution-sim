from living_agent_v2.discovery import arbitrate_hypothesis


def test_valid_hypothesis_passes_symbolic_arbitration():
    result = arbitrate_hypothesis("energy equals 500, mass equals 5, c equals 10")

    assert result.valid is True
    assert result.residual == 0


def test_invalid_hypothesis_fails_symbolic_arbitration():
    result = arbitrate_hypothesis("energy equals 501, mass equals 5, c equals 10")

    assert result.valid is False
    assert result.residual == 1


def test_missing_numeric_values_fails_arbitration():
    result = arbitrate_hypothesis("this material compresses reality")

    assert result.valid is False
    assert "energy and mass" in result.feedback

