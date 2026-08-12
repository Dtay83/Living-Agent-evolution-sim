from living_agent_v2.discovery import arbitrate_hypothesis


def test_accepts_mass_energy_equivalence() -> None:
    result = arbitrate_hypothesis("energy equals 500, mass equals 5, c equals 10")

    assert result.valid is True
    assert result.residual == 0.0


def test_rejects_invalid_equation() -> None:
    result = arbitrate_hypothesis("energy equals 499, mass equals 5, c equals 10")

    assert result.valid is False
    assert result.residual == -1.0


def test_rejects_missing_values_without_guessing() -> None:
    result = arbitrate_hypothesis("energy equals 500")

    assert result.valid is False
    assert result.residual is None
    assert "missing" in result.feedback


def test_rejects_nonphysical_inputs() -> None:
    result = arbitrate_hypothesis("energy equals 500, mass equals -5, c equals 10")

    assert result.valid is False
    assert result.residual is None
    assert "non-negative" in result.feedback
