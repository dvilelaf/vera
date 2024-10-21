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

"""This package contains round behaviours of VeraAbciApp."""

import json
import re
from abc import ABC
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Optional, Set, Tuple, Type, cast

from aea.protocols.base import Message
from textblob import TextBlob  # type: ignore
from twitter_text import parse_tweet  # type: ignore

from packages.dvilela.connections.genai.connection import (
    PUBLIC_ID as GENAI_CONNECTION_PUBLIC_ID,
)
from packages.dvilela.connections.kv_store.connection import (
    PUBLIC_ID as KV_STORE_CONNECTION_PUBLIC_ID,
)
from packages.dvilela.connections.twikit.connection import (
    PUBLIC_ID as TWIKIT_CONNECTION_PUBLIC_ID,
)
from packages.dvilela.protocols.kv_store.dialogues import (
    KvStoreDialogue,
    KvStoreDialogues,
)
from packages.dvilela.protocols.kv_store.message import KvStoreMessage
from packages.dvilela.skills.vera_abci.models import Params, SharedState
from packages.dvilela.skills.vera_abci.prompts import FACT_CHECK_PROMPT
from packages.dvilela.skills.vera_abci.rounds import (
    PostResponsesPayload,
    PostResponsesRound,
    SynchronizedData,
    UpdateNewsPoolPayload,
    UpdateNewsPoolRound,
    UpdateTweetsPoolPayload,
    UpdateTweetsPoolRound,
    VeraAbciApp,
)
from packages.valory.protocols.srr.dialogues import SrrDialogue, SrrDialogues
from packages.valory.protocols.srr.message import SrrMessage
from packages.valory.skills.abstract_round_abci.base import AbstractRound
from packages.valory.skills.abstract_round_abci.behaviours import (
    AbstractRoundBehaviour,
    BaseBehaviour,
)
from packages.valory.skills.abstract_round_abci.models import Requests


HTTP_OK = 200
MAX_TWEET_CHARS = 280
JSON_RESPONSE_REGEX = r"```json({.*})```"


def tweet_len(tweet: str) -> int:
    """Calculates a tweet length"""
    return parse_tweet(tweet).asdict()["weightedLength"]


class VeraBaseBehaviour(BaseBehaviour, ABC):  # pylint: disable=too-many-ancestors
    """Base behaviour for the vera_abci skill."""

    @property
    def synchronized_data(self) -> SynchronizedData:
        """Return the synchronized data."""
        return cast(SynchronizedData, super().synchronized_data)

    @property
    def params(self) -> Params:
        """Return the params."""
        return cast(Params, super().params)

    @property
    def local_state(self) -> SharedState:
        """Return the state."""
        return cast(SharedState, self.context.state)

    def _do_connection_request(
        self,
        message: Message,
        dialogue: Message,
        timeout: Optional[float] = None,
    ) -> Generator[None, None, Message]:
        """Do a request and wait the response, asynchronously."""

        self.context.outbox.put_message(message=message)
        request_nonce = self._get_request_nonce_from_dialogue(dialogue)  # type: ignore
        cast(Requests, self.context.requests).request_id_to_callback[
            request_nonce
        ] = self.get_callback_request()
        response = yield from self.wait_for_message(timeout=timeout)
        return response

    def _call_twikit(self, method: str, **kwargs: Any) -> Generator[None, None, Any]:
        """Send a request message from the skill context."""
        srr_dialogues = cast(SrrDialogues, self.context.srr_dialogues)
        srr_message, srr_dialogue = srr_dialogues.create(
            counterparty=str(TWIKIT_CONNECTION_PUBLIC_ID),
            performative=SrrMessage.Performative.REQUEST,
            payload=json.dumps({"method": method, "kwargs": kwargs}),
        )
        srr_message = cast(SrrMessage, srr_message)
        srr_dialogue = cast(SrrDialogue, srr_dialogue)
        response = yield from self._do_connection_request(srr_message, srr_dialogue)  # type: ignore

        response_json = json.loads(response.payload)  # type: ignore

        if "error" in response_json:
            self.context.logger.error(response_json["error"])
            return None

        return response_json["response"]  # type: ignore

    def _call_genai(
        self,
        prompt: str,
    ) -> Generator[None, None, Optional[str]]:
        """Send a request message from the skill context."""
        srr_dialogues = cast(SrrDialogues, self.context.srr_dialogues)
        srr_message, srr_dialogue = srr_dialogues.create(
            counterparty=str(GENAI_CONNECTION_PUBLIC_ID),
            performative=SrrMessage.Performative.REQUEST,
            payload=json.dumps({"prompt": prompt}),
        )
        srr_message = cast(SrrMessage, srr_message)
        srr_dialogue = cast(SrrDialogue, srr_dialogue)
        response = yield from self._do_connection_request(srr_message, srr_dialogue)  # type: ignore

        response_json = json.loads(response.payload)  # type: ignore

        if "error" in response_json:
            self.context.logger.error(response_json["error"])
            return None

        return response_json["response"]  # type: ignore

    def _read_kv(
        self,
        keys: Tuple[str],
    ) -> Generator[None, None, Optional[Dict]]:
        """Send a request message from the skill context."""
        self.context.logger.info(f"Reading keys from db: {keys}")
        kv_store_dialogues = cast(KvStoreDialogues, self.context.kv_store_dialogues)
        kv_store_message, srr_dialogue = kv_store_dialogues.create(
            counterparty=str(KV_STORE_CONNECTION_PUBLIC_ID),
            performative=KvStoreMessage.Performative.READ_REQUEST,
            keys=keys,
        )
        kv_store_message = cast(KvStoreMessage, kv_store_message)
        kv_store_dialogue = cast(KvStoreDialogue, srr_dialogue)
        response = yield from self._do_connection_request(
            kv_store_message, kv_store_dialogue  # type: ignore
        )
        if response.performative != KvStoreMessage.Performative.READ_RESPONSE:
            return None

        data = {key: response.data.get(key, None) for key in keys}  # type: ignore

        return data

    def _write_kv(
        self,
        data: Dict[str, str],
    ) -> Generator[None, None, bool]:
        """Send a request message from the skill context."""
        kv_store_dialogues = cast(KvStoreDialogues, self.context.kv_store_dialogues)
        kv_store_message, srr_dialogue = kv_store_dialogues.create(
            counterparty=str(KV_STORE_CONNECTION_PUBLIC_ID),
            performative=KvStoreMessage.Performative.CREATE_OR_UPDATE_REQUEST,
            data=data,
        )
        kv_store_message = cast(KvStoreMessage, kv_store_message)
        kv_store_dialogue = cast(KvStoreDialogue, srr_dialogue)
        response = yield from self._do_connection_request(
            kv_store_message, kv_store_dialogue  # type: ignore
        )
        return response == KvStoreMessage.Performative.SUCCESS

    def is_polarized_tweet(self, tweet: str) -> bool:
        """Checks whether a tweet is polarized"""
        analysis = TextBlob(tweet)
        # Polarity [-1, 1] and subjectivity [0, 1]  # noqa: E800
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity
        self.context.logger.info(f"polarity={polarity} subjectivity={subjectivity}")
        return polarity <= 0.5 and subjectivity >= 0.3

    def _get_utc_time(self) -> Any:
        """Get the current time"""
        now_utc = self.local_state.round_sequence.last_round_transition_timestamp

        # Tendermint timestamps are expected to be UTC, but for some reason
        # we are getting local time. We replace the hour and timezone.
        # TODO: this hour replacement could be problematic in some time zones
        now_utc = now_utc.replace(
            hour=datetime.now(timezone.utc).hour, tzinfo=timezone.utc
        )

        return now_utc


class UpdateNewsPoolBehaviour(VeraBaseBehaviour):  # pylint: disable=too-many-ancestors
    """TrackChainEventsBehaviour"""

    matching_round: Type[AbstractRound] = UpdateNewsPoolRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            fake_news = yield from self.get_updated_news()

            payload = UpdateNewsPoolPayload(
                sender=self.context.agent_address,
                fake_news=json.dumps(fake_news, sort_keys=True),
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    def get_updated_news(self) -> Generator[None, None, Dict]:
        """Get the latest fake news from the fact checker API"""

        fake_news = self.synchronized_data.fake_news

        params = {
            "languageCode": self.params.fact_checker_language,
            "query": self.params.fact_checker_query,
            "key": self.params.fact_checker_api_key,
            "maxAgeDays": self.params.fact_checker_max_days,
        }

        response = yield from self.get_http_response(  # type: ignore
            method="GET", url=self.params.fact_checker_url, parameters=params
        )

        if response.status_code != HTTP_OK:  # type: ignore
            self.context.logger.error(
                f"Error getting updated fake news: {response}"  # type: ignore
            )
            return fake_news

        # Add fresh claims
        response_json = json.loads(response.body)
        claims = response_json.get("claims", [])

        self.context.logger.info(f"Retrieved {len(claims)} fake news")
        for claim in claims:
            fake_news[claim["text"]] = claim

        return fake_news


class UpdateTweetsPoolBehaviour(
    VeraBaseBehaviour
):  # pylint: disable=too-many-ancestors
    """UpdateTweetsPoolBehaviour"""

    matching_round: Type[AbstractRound] = UpdateTweetsPoolRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            pending_tweets = yield from self.get_updated_tweets()

            payload = UpdateTweetsPoolPayload(
                sender=self.context.agent_address,
                pending_tweets=json.dumps(pending_tweets, sort_keys=True),
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    def get_updated_tweets(self) -> Generator[None, None, Dict]:
        """Get the latest tweets spreading fake news"""

        pending_tweets = self.synchronized_data.pending_tweets
        fake_news = self.synchronized_data.fake_news.keys()

        # Get responded tweets from the db
        response = yield from self._read_kv(keys=("responded_tweets",))

        if response is None:
            self.context.logger.error(
                "Error reading from the database. Responded tweets couldn't be loaded and therefore no new tweets will be searched."
            )
            return pending_tweets

        responded_tweets = (
            json.loads(response["responded_tweets"])
            if response["responded_tweets"]
            else []
        )
        self.context.logger.info(
            f"Loaded {len(responded_tweets)} responded tweets from db"
        )

        # Iterate all the fake news
        for fake_new in fake_news:
            self.context.logger.info(f"Searching tweets about {fake_new!r}")

            tweets = yield from self._call_twikit(method="search", query=fake_new)

            if tweets is None:
                self.context.logger.error("Could not get tweets for this fake new")
                continue

            self.context.logger.info(f"Retrieved {len(tweets)} tweets")

            # Iterate all the tweets
            for tweet in tweets:
                self.context.logger.info(
                    f"Analyzing tweet {tweet['id']!r}: {tweet['text']}"
                )

                if (
                    tweet["id"] in pending_tweets
                    and tweet["id"] not in responded_tweets
                ):
                    self.context.logger.info("Tweet was already processed")
                    continue

                tweet_time = datetime.strptime(
                    tweet["created_at"], "%a %b %d %H:%M:%S %z %Y"
                )

                if (
                    self._get_utc_time() - tweet_time
                ).total_seconds() > 250000:  # ~3 days
                    self.context.logger.info("Tweet is old")
                    continue

                # if not self.is_polarized_tweet(tweet["text"]):  # noqa: E800
                #     self.context.logger.info("Tweet is not polarized")  # noqa: E800
                #     continue  # noqa: E800

                is_popular = None

                try:
                    view_count = int(tweet["view_count"])
                    is_popular = view_count >= 500
                except Exception:  # nosec B110 # pylint: disable=broad-except
                    pass

                if is_popular is None:
                    try:
                        retweet_count = int(tweet["retweet_count"])
                        is_popular = retweet_count >= 20
                    except Exception:  # nosec B110 # pylint: disable=broad-except
                        pass

                if is_popular is None:
                    self.context.logger.info("Cannot determine the tweet's popularity")
                    continue

                if not is_popular:
                    self.context.logger.info(
                        f"Tweet is not very popular [view_count={view_count}]"
                    )
                    continue

                # Add the new tweet
                self.context.logger.info("New tweet added to the pool")
                pending_tweets[tweet["id"]] = tweet
                pending_tweets[tweet["id"]]["fake_new"] = fake_new

        return pending_tweets


class PostResponsesBehaviour(VeraBaseBehaviour):  # pylint: disable=too-many-ancestors
    """PrepareResponsesBehaviour"""

    matching_round: Type[AbstractRound] = PostResponsesRound

    def async_act(self) -> Generator:
        """Do the act, supporting asynchronous execution."""

        with self.context.benchmark_tool.measure(self.behaviour_id).local():
            pending_tweets = yield from self.post_responses()

            payload = PostResponsesPayload(
                sender=self.context.agent_address,
                pending_tweets=json.dumps(pending_tweets, sort_keys=True),
            )

        with self.context.benchmark_tool.measure(self.behaviour_id).consensus():
            yield from self.send_a2a_transaction(payload)
            yield from self.wait_until_round_end()

        self.set_done()

    def post_responses(  # pylint: disable=too-many-locals
        self,
    ) -> Generator[None, None, Dict]:
        """Post responses"""

        fake_news = self.synchronized_data.fake_news
        pending_tweets = self.synchronized_data.pending_tweets

        # Get responded tweets from the db
        response = yield from self._read_kv(keys=("responded_tweets",))

        if response is None:
            self.context.logger.error(
                "Error reading from the database. Responded tweets couldn't be loaded and therefore no new tweets will be responded."
            )
            return pending_tweets

        responded_tweets = (
            json.loads(response["responded_tweets"])
            if response["responded_tweets"]
            else []
        )
        self.context.logger.info(
            f"Loaded {len(responded_tweets)} responded tweets from db"
        )

        self.context.logger.info(
            f"There are {len(pending_tweets.values())} tweets pending to be responded"
        )

        # Prepare responses
        n_tweets_this_period = 0

        for tweet in pending_tweets.values():
            if n_tweets_this_period >= self.params.max_tweets_per_period:
                self.context.logger.info("Reached max tweets responded this period")
                break

            self.context.logger.info(
                f"Preparing response for tweet {tweet['id']} related to: {tweet['fake_new']}"
            )

            fake_new = fake_news[tweet["fake_new"]]
            claim_review = fake_new["claimReview"][0]  # Get the first review only
            prompt = FACT_CHECK_PROMPT.format(
                title=tweet["fake_new"],
                claimer=claim_review["publisher"]["name"],
                rating=claim_review["textualRating"],
                url=claim_review["url"],
                tweet=tweet["text"],
            )
            llm_response = yield from self._call_genai(prompt=prompt)
            self.context.logger.info(f"LLM response: {llm_response}")

            if llm_response is None:
                continue

            # Postprocess response
            llm_response = llm_response.replace("\n", "").strip()
            match = re.match(JSON_RESPONSE_REGEX, llm_response)

            if match:
                response_json = json.loads(match.groups()[0])
            else:
                response_json = json.loads(llm_response)

            is_fake = response_json.get("is_fake", False)
            response_tweet = response_json.get("response_tweet", None)

            if not is_fake or not response_tweet:
                self.context.logger.info(
                    "Tweet is not fake news or response format is not parseable"
                )
                continue

            t_len = tweet_len(response_tweet)
            if t_len > MAX_TWEET_CHARS:
                self.context.logger.error(
                    f"Tweet is too long [{t_len}]. will retry later: {response_tweet}"
                )
                continue

            self.context.logger.info(
                f"Response to tweet {tweet['id']} is OK!: {response_tweet}"
            )

            if not self.params.enable_posting:
                self.context.logger.info("Posting is disabled")
                continue

            self.context.logger.info("Posting the response...")

            tweet_ids = yield from self._call_twikit(
                method="post",
                tweets=[{"text": response_tweet, "reply_to": tweet["id"]}],
            )

            if tweet_ids is None:
                continue

            n_tweets_this_period += 1
            responded_tweets.append(tweet_ids[0])

        # Write responded tweets
        yield from self._write_kv({"responded_tweets": json.dumps(responded_tweets)})
        self.context.logger.info("Wrote responded_tweets to db")

        # Remove pending tweets
        pending_tweets = {
            k: v for k, v in pending_tweets.items() if k not in responded_tweets
        }

        return pending_tweets


class VeraRoundBehaviour(AbstractRoundBehaviour):
    """VeraRoundBehaviour"""

    initial_behaviour_cls = UpdateNewsPoolBehaviour
    abci_app_cls = VeraAbciApp  # type: ignore
    behaviours: Set[Type[BaseBehaviour]] = [  # type: ignore
        UpdateNewsPoolBehaviour,
        UpdateTweetsPoolBehaviour,
        PostResponsesBehaviour,
    ]
