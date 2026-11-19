"""Decentralized taxi-positioning game: model, best-response dynamics, equilibrium enumeration."""
from game.model import TaxiGame
from game.dynamics import best_response_dynamics, BRDResult, initial_profile
from game.enumeration import enumerate_equilibria, EnumerationResult
from game.instance import InstanceConfig, build_game, run_instance

__all__ = ["TaxiGame", "best_response_dynamics", "BRDResult", "initial_profile",
           "enumerate_equilibria", "EnumerationResult", "InstanceConfig", "build_game", "run_instance"]
