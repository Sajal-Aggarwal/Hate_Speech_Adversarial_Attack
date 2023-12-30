"""
DeepWordBug
========================================
(Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers)
"""

from textattack import Attack
from textattack.constraints.overlap import LevenshteinEditDistance
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import (
    CompositeTransformation,
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
    WordSwapRandomCharacterSubstitution,
)

from .attack_recipe import AttackRecipe

class DeepWordBugGao2018(AttackRecipe):
    @staticmethod
    def build(model_wrapper, use_all_transformations=True):
        # Swap characters out from words. Choose the best of four potential transformations.
        if use_all_transformations:
            transformation = CompositeTransformation(
                [
                    # (1) Swap: Swap two adjacent letters in the word.
                    WordSwapNeighboringCharacterSwap(),
                    # (2) Substitution: Substitute a letter in the word with a random letter.
                    WordSwapRandomCharacterSubstitution(),
                    # (3) Deletion: Delete a random letter from the word.
                    WordSwapRandomCharacterDeletion(),
                    # (4) Insertion: Insert a random letter in the word.
                    WordSwapRandomCharacterInsertion(),
                ]
            )
        else:
            # Using the Combined Score and the Substitution Transformer to generate
            # adversarial samples, with the maximum edit distance difference of 30
            # (ϵ = 30).
            transformation = WordSwapRandomCharacterSubstitution()
        constraints = [RepeatModification(), StopwordModification()]
        #
        # Maximum difference on edit distance (ϵ) is 30 for each sample.
        #
        constraints.append(LevenshteinEditDistance(30))
        #
        # Goal is untargeted classification
        #
        goal_function = UntargetedClassification(model_wrapper)
        #
        # Greedily swapping words with "Word Importance Ranking".
        #
        search_method = GreedyWordSwapWIR()

        return Attack(goal_function, constraints, transformation, search_method)
