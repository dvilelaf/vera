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

"""This package contains the rounds of VeraAbciApp."""

import json
from enum import Enum
from typing import Dict, FrozenSet, Set, cast

from packages.dvilela.skills.vera_abci.payloads import (
    PostResponsesPayload,
    UpdateNewsPoolPayload,
    UpdateTweetsPoolPayload,
)
from packages.valory.skills.abstract_round_abci.base import (
    AbciApp,
    AbciAppTransitionFunction,
    AppState,
    BaseSynchronizedData,
    CollectSameUntilThresholdRound,
    CollectionRound,
    DegenerateRound,
    DeserializedCollection,
    EventToTimeout,
    get_name,
)


class Event(Enum):
    """VeraAbciApp Events"""

    ERROR = "error"
    DONE = "done"
    NO_MAJORITY = "no_majority"
    ROUND_TIMEOUT = "round_timeout"
    RETRY = "retry"


class SynchronizedData(BaseSynchronizedData):
    """
    Class to represent the synchronized data.

    This data is replicated by the tendermint application.
    """

    def _get_deserialized(self, key: str) -> DeserializedCollection:
        """Strictly get a collection and return it deserialized."""
        serialized = self.db.get_strict(key)
        return CollectionRound.deserialize_collection(serialized)

    @property
    def fake_news(self) -> dict:
        """Get the fake news."""
        return cast(dict, json.loads(cast(str, self.db.get("fake_news", "{}"))))

    @property
    def participant_to_news(self) -> DeserializedCollection:
        """Get the participants to the news update round."""
        return self._get_deserialized("participant_to_news")

    @property
    def pending_tweets(self) -> dict:
        """Get the pending tweets."""
        return cast(dict, json.loads(cast(str, self.db.get("pending_tweets", "{}"))))

    @property
    def participant_to_tweets(self) -> DeserializedCollection:
        """Get the participants to the news update round."""
        return self._get_deserialized("participant_to_tweets")


class UpdateNewsPoolRound(CollectSameUntilThresholdRound):
    """UpdateNewsPoolRound"""

    payload_class = UpdateNewsPoolPayload
    synchronized_data_class = SynchronizedData
    done_event = Event.DONE
    no_majority_event = Event.NO_MAJORITY
    collection_key = get_name(SynchronizedData.participant_to_news)
    selection_key = get_name(SynchronizedData.fake_news)

    # Event.ROUND_TIMEOUT  # this needs to be mentioned for static checkers


class UpdateTweetsPoolRound(CollectSameUntilThresholdRound):
    """UpdateTweetsPoolRound"""

    payload_class = UpdateTweetsPoolPayload
    synchronized_data_class = SynchronizedData
    done_event = Event.DONE
    no_majority_event = Event.NO_MAJORITY
    collection_key = get_name(SynchronizedData.participant_to_tweets)
    selection_key = get_name(SynchronizedData.pending_tweets)

    # Event.ROUND_TIMEOUT  # this needs to be mentioned for static checkers


class PostResponsesRound(CollectSameUntilThresholdRound):
    """PostResponsesRound"""

    payload_class = PostResponsesPayload
    synchronized_data_class = SynchronizedData
    done_event = Event.DONE
    no_majority_event = Event.NO_MAJORITY
    collection_key = get_name(SynchronizedData.participant_to_tweets)
    selection_key = get_name(SynchronizedData.pending_tweets)

    # Event.ROUND_TIMEOUT  # this needs to be mentioned for static checkers


class FinishedPublishRound(DegenerateRound):
    """FinishedPublishRound"""


class VeraAbciApp(AbciApp[Event]):
    """VeraAbciApp"""

    initial_round_cls: AppState = UpdateNewsPoolRound
    initial_states: Set[AppState] = {UpdateNewsPoolRound}
    transition_function: AbciAppTransitionFunction = {
        UpdateNewsPoolRound: {
            Event.DONE: UpdateTweetsPoolRound,
            Event.NO_MAJORITY: UpdateNewsPoolRound,
            Event.ROUND_TIMEOUT: UpdateNewsPoolRound,
        },
        UpdateTweetsPoolRound: {
            Event.DONE: PostResponsesRound,
            Event.NO_MAJORITY: UpdateTweetsPoolRound,
            Event.ROUND_TIMEOUT: UpdateTweetsPoolRound,
        },
        PostResponsesRound: {
            Event.DONE: FinishedPublishRound,
            Event.NO_MAJORITY: UpdateTweetsPoolRound,
            Event.ROUND_TIMEOUT: UpdateTweetsPoolRound,
        },
        FinishedPublishRound: {},
    }
    final_states: Set[AppState] = {FinishedPublishRound}
    event_to_timeout: EventToTimeout = {}
    cross_period_persisted_keys: FrozenSet[str] = frozenset()
    db_pre_conditions: Dict[AppState, Set[str]] = {
        UpdateNewsPoolRound: set(),
        UpdateTweetsPoolRound: set(),
        PostResponsesRound: set(),
    }
    db_post_conditions: Dict[AppState, Set[str]] = {
        FinishedPublishRound: set(),
    }
