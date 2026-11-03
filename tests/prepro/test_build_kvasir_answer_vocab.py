from harness.prepro.build_kvasir_answer_vocab import apply_vocab, build_vocab


def test_build_vocab_assigns_stable_indices_by_frequency():
    answers = [["polyp"], ["polyp"], ["polyp"], ["ulcerative colitis"], ["ulcerative colitis"], ["yes"]]
    vocab = build_vocab(answers)
    assert vocab["polyp"] == 0  # most frequent first
    assert vocab["ulcerative colitis"] == 1
    assert vocab["yes"] == 2
    assert len(vocab) == 3


def test_build_vocab_breaks_ties_alphabetically():
    answers = [["b"], ["a"]]  # tie in frequency (1 each)
    vocab = build_vocab(answers)
    assert vocab["a"] == 0
    assert vocab["b"] == 1


def test_apply_vocab_maps_answers_to_labels_and_unknown_to_minus_one():
    vocab = {"polyp": 0, "yes": 1}
    labels = apply_vocab([["polyp"], ["yes"], ["never seen"]], vocab)
    assert labels == [[0], [1], [-1]]
