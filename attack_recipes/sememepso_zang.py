"""
Particle Swarm Optimization
==================================
(Word-level Textual Adversarial Attacking as Combinatorial Optimization)
"""
from textattack import Attack
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import ParticleSwarmOptimization
from textattack.transformations import WordSwapHowNet

from .attack_recipe import AttackRecipe

class PSOZang2020(AttackRecipe):
    @staticmethod
    def build(model_wrapper):
        # Swapping words with their synonyms extracted based on the HowNet
        transformation = WordSwapHowNet()
        constraints = [RepeatModification(), StopwordModification()]
        # During entailment, only hypothesis is edited - keeping premise the same.
        #
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        goal_function = UntargetedClassification(model_wrapper)
        # Performing word substitution with PSO
        search_method = ParticleSwarmOptimization(pop_size=60, max_iters=20)

        return Attack(goal_function, constraints, transformation, search_method)
