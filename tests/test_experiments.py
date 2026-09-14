import pytest

from sperner.experiments import (
    DivideAndChoose,
    SpernerRefinement,
    SpernerSingleWalk,
    SplidditMaximin,
    main,
    random_flats,
    run,
)
from sperner.people import Budgeted, QuasiLinear


def test_random_flats_are_reproducible_and_values_add_up_to_the_rent():
    flats = random_flats(3, 5, seed=7)
    assert flats == random_flats(3, 5, seed=7)
    assert flats != random_flats(3, 5, seed=8)
    for flat in flats:
        assert flat.n == 3
        for person in flat.people:
            assert isinstance(person, QuasiLinear)
            assert sum(person.values) == pytest.approx(3000)


def test_flats_with_budgets():
    for flat in random_flats(3, 5, budgets=(0.3, 0.4)):
        for person in flat.people:
            assert isinstance(person, Budgeted)
            assert 900 <= person.budget <= 1200


@pytest.mark.parametrize("n", [2, 3])
def test_methods_stay_within_their_guarantee(n):
    methods = [SpernerRefinement(), SpernerRefinement(factor=2), SpernerSingleWalk()]
    if n == 2:
        methods.append(DivideAndChoose())
    results = run(random_flats(n, 15, seed=1), methods, tolerance=10)
    answered = [row for row in results.rows if row.rests_on_answers]
    assert len(answered) >= 0.9 * len(results.rows)
    for row in answered:
        assert row.envy <= row.guarantee + 1e-6


def test_refinement_needs_far_fewer_questions_than_a_single_walk():
    results = run(random_flats(3, 10, seed=2), [SpernerRefinement(), SpernerSingleWalk()], 10)
    mean = {s["method"]: s["inputs_mean"] for s in results.summary()}
    assert mean["sperner"] < mean["single walk"] / 2


def test_divide_and_choose_is_for_two_flatmates_only():
    assert run(random_flats(3, 3), [DivideAndChoose()], 10).rows == []
    rows = run(random_flats(2, 3), [DivideAndChoose()], 10).rows
    assert len(rows) == 3
    # The first answers log2(3000 / 10), about 8 questions, and the second one.
    assert all(row.inputs <= 7 for row in rows)


def test_spliddit_is_exact():
    pytest.importorskip("scipy")
    results = run(random_flats(3, 5), [SplidditMaximin()], 10)
    assert all(row.envy == pytest.approx(0, abs=1e-6) for row in results.rows)
    assert "exact" in results.markdown()


def test_custom_methods_without_applies_run_everywhere():
    class Everyone:
        name = "all at the same price"

        def __call__(self, flat, tolerance):
            from sperner.experiments import Outcome

            price = flat.rent / flat.n
            return Outcome(tuple(range(flat.n)), (price,) * flat.n, (0,) * flat.n, None)

    assert len(run(random_flats(2, 4), [Everyone()], 10).rows) == 4


def test_markdown_and_csv(tmp_path):
    results = run(random_flats(2, 4), [SpernerRefinement()], 30)
    assert results.markdown().splitlines()[2].startswith("| sperner | 2 | 30 | 4 |")
    path = tmp_path / "rows.csv"
    results.to_csv(str(path))
    assert len(path.read_text().splitlines()) == 5


def test_command_line(capsys):
    main(["--people", "2", "--flats", "3", "--tolerance", "50"])
    out = capsys.readouterr().out
    assert "| sperner | 2 | 50 | 3 |" in out
    assert "| divide and choose | 2 | 50 | 3 |" in out


def test_the_tolerance_must_be_positive():
    for tolerance in (0, -5):
        with pytest.raises(ValueError, match="positive"):
            run(random_flats(2, 1), [DivideAndChoose()], tolerance)


def test_divide_and_choose_stops_when_floats_run_out():
    rows = run(random_flats(2, 1), [DivideAndChoose()], 1e-13).rows
    assert len(rows) == 1 and rows[0].inputs < 60


def test_a_method_with_a_bound_that_happens_to_be_exact_is_not_called_exact():
    pytest.importorskip("scipy")
    from sperner.experiments import Outcome

    class Bounded:
        name = "bounded"

        def __call__(self, flat, tolerance):
            exact = SplidditMaximin()(flat, tolerance)
            return Outcome(exact.assignment, exact.prices, exact.inputs, guarantee=tolerance)

    table = run(random_flats(2, 3), [Bounded()], 10).markdown()
    assert "| 0.00 |" in table.splitlines()[2]
