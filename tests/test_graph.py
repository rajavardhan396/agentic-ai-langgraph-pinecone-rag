from app.graph import grade_node

def test_empty_context_has_zero_confidence():
    assert grade_node({"context": [], "answer": "anything", "query": "q"})["confidence"] == 0.0
