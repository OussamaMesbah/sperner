from webapp.voting import PROFILES, RULES, borda, majority, order, violations


def test_condorcet_cycle():
    assert majority((("A", "B", "C"), ("B", "C", "A"), ("C", "A", "B"))) is None
    assert order(borda((("A", "B", "C"),) * 3)) == "A > B > C"


def test_each_rule_breaks_what_arrow_says_it_must():
    broken = {name: {v.axiom for v in violations(rule)} for name, rule in RULES.items()}
    assert broken["Majority"] == {"a ranking for every profile"}
    assert broken["Borda count"] == {"independence of irrelevant alternatives"}
    assert "independence of irrelevant alternatives" in broken["Plurality"]
    assert broken["Voter 1 decides"] == {"no dictator"}
    assert all(broken.values())  # Arrow: every rule breaks something


def test_the_examples_really_break_independence():
    [violation] = [v for v in violations(RULES["Borda count"]) if v.profiles]
    first, second = violation.profiles
    assert len(PROFILES) == 216 and first != second
