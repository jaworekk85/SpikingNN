"""Declared, swappable learning rules."""

from .base import LearningRule, LocalObservation, linear_synapses, local_observation
from .declaration import LOCALITIES, MODULATOR_SOURCES, RuleDeclaration
from .precision import constrain_
from .stdp import STDP, STDPConfig, stdp_update
from .surrogate_bptt import SurrogateBPTT, SurrogateBPTTConfig
from .three_factor import ThreeFactor, ThreeFactorConfig, eligibility_traces, three_factor_update
