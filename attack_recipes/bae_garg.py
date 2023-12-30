"""
BAE (BAE: BERT-Based Adversarial Examples)
============================================
"""
from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapMaskedLM

from .attack_recipe import AttackRecipe


class BAEGarg2019(AttackRecipe):
    """
    This is "attack mode" 1 from the paper, BAE-R, word replacement.

    There are 4 attack modes for BAE based on the
        R and I operations, where for each token t in S:
        • BAE-R: Replace token t (See Algorithm 1)
        • BAE-I: Insert a token to the left or right of t
        • BAE-R/I: Either replace token t or insert a
        token to the left or right of t
        • BAE-R+I: First replace token t, then insert a
        token to the left or right of t
    """

    @staticmethod
    def build(model_wrapper):
        transformation = WordSwapMaskedLM(
            method="bae", max_candidates=50, min_confidence=0.0
        )
        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        use_constraint = UniversalSentenceEncoder(
            threshold=0.936338023,
            metric="cosine",
            compare_against_original=True,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)
        goal_function = UntargetedClassification(model_wrapper)
        search_method = GreedyWordSwapWIR(wir_method="delete")

        return BAEGarg2019(goal_function, constraints, transformation, search_method)
