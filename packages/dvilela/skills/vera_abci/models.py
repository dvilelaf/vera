# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2024 David Vilela Freire
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------

"""This module contains the shared state for the abci skill of VeraAbciApp."""

from typing import Any

from packages.dvilela.skills.vera_abci.rounds import VeraAbciApp
from packages.valory.skills.abstract_round_abci.models import ApiSpecs, BaseParams
from packages.valory.skills.abstract_round_abci.models import (
    BenchmarkTool as BaseBenchmarkTool,
)
from packages.valory.skills.abstract_round_abci.models import Requests as BaseRequests
from packages.valory.skills.abstract_round_abci.models import (
    SharedState as BaseSharedState,
)


class SharedState(BaseSharedState):
    """Keep the current shared state of the skill."""

    abci_app_cls = VeraAbciApp


Requests = BaseRequests
BenchmarkTool = BaseBenchmarkTool


class RandomnessApi(ApiSpecs):
    """A model that wraps ApiSpecs for randomness api specifications."""


class Params(BaseParams):  # pylint: disable=too-many-instance-attributes
    """Parameters."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the parameters object."""
        self.fact_checker_url = self._ensure("fact_checker_url", kwargs, str)
        self.fact_checker_language = self._ensure("fact_checker_language", kwargs, str)
        self.fact_checker_query = self._ensure("fact_checker_query", kwargs, str)
        self.fact_checker_api_key = self._ensure("fact_checker_api_key", kwargs, str)
        self.fact_checker_max_days = self._ensure("fact_checker_max_days", kwargs, int)
        self.service_endpoint = self._ensure("service_endpoint", kwargs, str)
        self.enable_posting = self._ensure("enable_posting", kwargs, bool)
        self.max_tweets_per_period = self._ensure("max_tweets_per_period", kwargs, int)

        super().__init__(*args, **kwargs)
