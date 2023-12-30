"""
Pruthi2019: Combating with Robust Word Recognition
=================================================================
"""
from textattack import Attack
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import (
    MinWordLength,
    RepeatModification,
    StopwordModification,
)
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedySearch
from textattack.transformations import (
    CompositeTransformation,
    WordSwapNeighboringCharacterSwap,
    WordSwapQWERTY,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
)

from .attack_recipe import AttackRecipe

class Pruthi2019(AttackRecipe):
    @staticmethod
    def build(model_wrapper, max_num_word_swaps=1):
        # a combination of 4 different character-based transforms
        # ignore the first and last letter of each word, as in the paper
        transformation = CompositeTransformation(
            [
                WordSwapNeighboringCharacterSwap(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapRandomCharacterDeletion(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapRandomCharacterInsertion(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
                WordSwapQWERTY(
                    random_one=False, skip_first_char=True, skip_last_char=True
                ),
            ]
        )
        # Only editing words of length >= 4, edit max_num_word_swaps words.
        # Not editing the same word twice, so
        # max_num_word_swaps is the max number of character changes that can be made.
        constraints = [
            MinWordLength(min_length=4),
            StopwordModification(),
            MaxWordsPerturbed(max_num_words=max_num_word_swaps),
            RepeatModification(),
        ]
        # untargeted attack
        goal_function = UntargetedClassification(model_wrapper)
        search_method = GreedySearch()
        return Attack(goal_function, constraints, transformation, search_method)
