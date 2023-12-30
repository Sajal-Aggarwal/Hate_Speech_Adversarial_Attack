"""
TextBugger
===============
(TextBugger: Generating Adversarial Text Against Real-world Applications)
"""

from textattack import Attack
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import (
    CompositeTransformation,
    WordSwapEmbedding,
    WordSwapHomoglyphSwap,
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
)

from .attack_recipe import AttackRecipe
class TextBuggerLi2018(AttackRecipe):
    @staticmethod
    def build(model_wrapper):
        transformation = CompositeTransformation(
            [
                # (1) Insert: Inserting a space into the word
                WordSwapRandomCharacterInsertion(
                    random_one=True,
                    letters_to_insert=" ",
                    skip_first_char=True,
                    skip_last_char=True,
                ),
                # (2) Delete: Deleting a random character of the word except for the first and the last character
                WordSwapRandomCharacterDeletion(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                # (3) Swap: Swapping random two adjacent letters in the word without altering the first or last letter.
                WordSwapNeighboringCharacterSwap(
                    random_one=True, skip_first_char=True, skip_last_char=True
                ),
                # (4) Substitute-W: Replacing a word with its topk nearest neighbors in a context-aware word vector space. 
                WordSwapEmbedding(max_candidates=5),
            ]
        )

        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(UniversalSentenceEncoder(threshold=0.8))
        goal_function = UntargetedClassification(model_wrapper)
        search_method = GreedyWordSwapWIR(wir_method="delete")
        return Attack(goal_function, constraints, transformation, search_method)
