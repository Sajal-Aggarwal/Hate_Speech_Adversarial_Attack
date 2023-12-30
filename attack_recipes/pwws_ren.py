"""
PWWS
=======
(Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency)
"""
from textattack import Attack
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
)
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR
from textattack.transformations import WordSwapWordNet

from .attack_recipe import AttackRecipe

class PWWSRen2019(AttackRecipe):
    @staticmethod
    def build(model_wrapper):
        transformation = WordSwapWordNet()
        constraints = [RepeatModification(), StopwordModification()]
        goal_function = UntargetedClassification(model_wrapper)
        # searching over words based on a combination of their saliency score, and how efficient the WordSwap transform is
        search_method = GreedyWordSwapWIR("weighted-saliency")
        return Attack(goal_function, constraints, transformation, search_method)
