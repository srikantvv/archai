# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Type

from overrides import overrides

from archai.supergraph.nas.exp_runner import ExperimentRunner
from archai.supergraph.nas.arch_trainer import TArchTrainer
from archai.supergraph.algos.xnas.xnas_model_desc_builder import XnasModelDescBuilder
from archai.supergraph.algos.xnas.xnas_arch_trainer import XnasArchTrainer

class XnasExperimentRunner(ExperimentRunner):
    @overrides
    def model_desc_builder(self)->XnasModelDescBuilder:
        return XnasModelDescBuilder()

    @overrides
    def trainer_class(self)->TArchTrainer:
        return XnasArchTrainer

