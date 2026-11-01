from make_dataset import parse_call
from src import features as F


def _turns():
    return [
        {"speaker": "Operator", "text": "Welcome. We will take questions after the remarks."},
        {"speaker": "Jane CEO", "text": "We delivered strong growth and excellent results."},
        {"speaker": "Bob CFO", "text": "Margins improved on efficiency."},
        {"speaker": "Operator", "text": "We will now begin the question-and-answer session."},
        {"speaker": "Ann Analyst", "text": "Can you explain the weak decline in demand?"},
        {"speaker": "Jane CEO", "text": "I think it is too early to say."},
    ]


def test_roles_and_split():
    p = parse_call(_turns())
    assert p["has_qa"]
    assert p["roles"] == {"Ann Analyst": "analyst", "Jane CEO": "management"}
    assert "weak decline" in p["analyst_qa"] and "too early" in p["mgmt_qa"]
    assert "too early" not in p["analyst_qa"]


def test_intro_mention_of_questions_is_ignored():
    p = parse_call(_turns())
    assert p["text"].index("[[QA_START]]") > p["text"].index("Bob CFO")


def test_no_qa_detected():
    p = parse_call([{"speaker": "Jane CEO", "text": "Hello."}])
    assert not p["has_qa"] and p["analyst_qa"] == ""


def test_short_role_text_is_nan():
    import numpy as np
    assert np.isnan(F.role_features("too short", "mgmt")["mgmt_pos"])


def test_qa_only_executive_is_management_not_analyst():
    turns = _turns()[:4] + [
        {"speaker": "Ann Analyst", "text": "Can you explain the weak decline in demand?"},
        {"speaker": "Dan Divisionhead", "text": "Demand softened in one region and we expect it to recover."},
        {"speaker": "Operator", "text": "Our next question comes from Zed Analyst."},
        {"speaker": "Zed Analyst", "text": "Thanks for the color on margins."},
        {"speaker": "Jane CEO", "text": "Margins should be stable."},
    ]
    roles = parse_call(turns)["roles"]
    assert roles["Dan Divisionhead"] == "management"
    assert roles["Ann Analyst"] == "analyst"
    assert roles["Zed Analyst"] == "analyst"
    assert roles["Jane CEO"] == "management"
