"""Stage 2 — phonology-aware augmented edit distance between Arabic word forms.

``precompute_probabilities`` builds the CODA/CAPHI/ORTHO probability tables,
``substitution_weight`` turns them into a character substitution cost, and
``augmented_edit_distance`` wraps that cost in an affix-aware DP.
"""
