"""Word-level Levenshtein counts for corpus-aggregated greedy CTC WER."""


def word_error_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    reference = reference.lower().split()
    hypothesis = hypothesis.lower().split()
    previous = list(range(len(hypothesis) + 1))
    for index, expected in enumerate(reference, 1):
        current = [index]
        for column, actual in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (expected != actual)))
        previous = current
    return previous[-1], len(reference)


def word_error_rate(errors: int, words: int) -> float:
    # Empty reference: 0 for silence, otherwise count inserted words.
    return errors / max(1, words)
