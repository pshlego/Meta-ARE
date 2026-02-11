# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import json
import logging
from typing import Iterator

from tqdm import tqdm

from are.simulation.data_handler.models import ExportedHuggingFaceMetadata
from are.simulation.scenarios.scenario_imported_from_json.benchmark_scenario import (
    BenchmarkScenarioImportedFromJson,
)
from are.simulation.types import CompletedEvent
from are.simulation.utils.countable_iterator import CountableIterator

logger: logging.Logger = logging.getLogger(__name__)


def load_oracle_data_mapping(
    dataset_name: str,
    dataset_config: str,
    dataset_split: str,
    dataset_revision: str | None = None,
) -> dict[str, dict]:
    """Load oracle data from a HuggingFace dataset and create a mapping of scenario_id to oracle data.

    This function loads a HuggingFace dataset containing oracle data and extracts oracle events,
    hints, and completed events for each scenario. It creates a mapping from scenario_id to
    the extracted oracle data.

    Oracle datasets only contain base configs (e.g., 'execution', 'search') and not artificial
    configs created by gaia2-run (e.g., 'execution_a2a', 'execution_noise'). This function
    automatically maps artificial configs back to their base configs when loading oracle data.

    :param dataset_name: The name of the HuggingFace dataset containing oracle data
    :param dataset_config: The config (subset) of the dataset (may be artificial config like 'execution_a2a')
    :param dataset_split: The split of the dataset (e.g., "test", "validation", "train")
    :param dataset_revision: Optional revision of the dataset
    :return: A mapping of scenario_id to oracle data containing hints, oracle events, and completed events
    :rtype: dict[str, dict]
    :raises ValueError: If the base config is not found in the oracle dataset
    """

    from datasets import IterableDatasetDict, load_dataset

    oracle_data_mapping = {}

    # Load with the base config (oracle datasets only have base configs)
    ds = load_dataset(
        dataset_name,
        name=dataset_config,
        dataset_revision=dataset_revision,
        streaming=True,
    )

    assert isinstance(ds, IterableDatasetDict), (
        f"Looks like the {dataset_name} is not split by sets."
    )
    assert dataset_split in ds, (
        f"Split '{dataset_split}' not found in dataset '{dataset_name}/{dataset_config}'. Should be a split of the dataset."
    )

    logger.info(
        f"Loading oracle data from {dataset_name}/{dataset_config}/{dataset_split}"
    )
    for row in ds[dataset_split]:
        scenario_id = row.get("scenario_id")
        if not scenario_id:
            continue

        # Parse the data to extract oracle events and hints
        try:
            data = json.loads(row["data"])

            # Extract oracle events
            oracle_events = []
            if "events" in data:
                oracle_events = [
                    event
                    for event in data["events"]
                    if event.get("class_name") == "OracleEvent"
                ]

            # Extract hints
            hints = []
            if (
                "metadata" in data
                and "definition" in data["metadata"]
                and "hints" in data["metadata"]["definition"]
            ):
                hints = data["metadata"]["definition"]["hints"]

            # Extract completed_events
            completed_events = []
            if "completed_events" in data:
                completed_events = data["completed_events"]

            apps = []
            if "apps" in data:
                apps = data["apps"]

            # Store the oracle data
            oracle_data_mapping[scenario_id] = {
                "oracle_events": oracle_events,
                "hints": hints,
                "completed_events": completed_events,
                "apps": apps,
            }
        except Exception as e:
            logger.warning(
                f"Failed to parse oracle data for scenario {scenario_id}: {str(e)}"
            )
            continue

    logger.info(f"Loaded oracle data for {len(oracle_data_mapping)} scenarios")
    return oracle_data_mapping


def _create_huggingface_scenario_iterator(
    dataset_name: str,
    dataset_config: str,
    dataset_split: str = "validation",
    dataset_revision: str | None = None,
    load_completed_events: bool = False,
    limit: int | None = None,
    **kwargs,
) -> tuple[
    Iterator[tuple[BenchmarkScenarioImportedFromJson, list[CompletedEvent]]], int | None
]:
    """Create an iterator over the scenarios in a HuggingFace dataset.

    This function loads scenarios from a HuggingFace dataset and yields them one by one.
    It can optionally merge oracle data from another dataset if specified.

    The oracle dataset config is automatically derived from the dataset_config by mapping
    artificial configs (like 'execution_a2a', 'execution_noise') back to their base configs
    (like 'execution') since oracle datasets only contain base configs.

    :param dataset_name: The name of the HuggingFace dataset
    :param dataset_config: The config (subset) of the dataset (e.g., "ambiguity", "execution", "time", "execution_a2a", "execution_noise")
    :param dataset_split: The split of the dataset (e.g., "test", "validation", "train"), defaults to "test"
    :param dataset_revision: Optional revision of the dataset
    :param load_completed_events: Whether to load completed events for offline validation
    :param limit: Maximum number of scenarios to load, if None loads all scenarios
    :return: Tuple of (iterator of tuples containing (scenario, completed_events), total count or None)
    :rtype: tuple[Iterator[tuple[BenchmarkScenarioImportedFromJson, list[CompletedEvent]]], int | None]
    :raises ValueError: If oracle dataset is specified but a scenario is not found in it
    """

    from datasets import IterableDatasetDict, load_dataset

    from are.simulation.benchmark.scenario_loader import load_scenario

    oracle_data_mapping = {}  # noqa: F841

    try:
        # Load dataset with config (subset) parameter
        ds = load_dataset(
            dataset_name,
            name=dataset_config,
            dataset_revision=dataset_revision,
            streaming=True,
        )
        assert isinstance(ds, IterableDatasetDict), (
            f"Looks like the {dataset_name} is not split by sets."
        )
        assert dataset_split in ds, (
            f"Split '{dataset_split}' not found in dataset '{dataset_name}/{dataset_config}'. Should be a split of the dataset."
        )

        # Get the total count of examples in the dataset
        total_count = None
        try:
            # Try to get the number of examples from the dataset info
            splits = ds[dataset_split].info.splits
            if splits is not None and dataset_split in splits:
                total_count = splits[dataset_split].num_examples
                logger.info(
                    f"Dataset has {total_count} examples in split {dataset_split}"
                )
            else:
                logger.warning(
                    f"Dataset info splits is None or doesn't contain {dataset_split}"
                )
        except (AttributeError, KeyError) as e:
            logger.warning(f"Could not determine total count from dataset info: {e}")

        # Apply limit if specified
        if limit is not None:
            logger.info(f"Limiting to {limit} scenarios from HuggingFace dataset")
            total_count = min(limit, total_count) if total_count is not None else limit

        def scenario_generator():
            # Wrap the iterator with tqdm to show progress
            for index, row in enumerate(
                tqdm(
                    ds[dataset_split],
                    desc="Loading scenarios from HuggingFace",
                    position=0,
                    leave=True,
                    total=total_count,
                ),
                start=1,
            ):
                # Stop if we've reached the limit
                if limit is not None and index > limit:
                    logger.info(f"Reached limit of {limit} scenarios")
                    break
                scenario_id = row.get("scenario_id")
                if not scenario_id:
                    logger.warning(f"Skipping row {index} due to missing scenario_id")
                    continue

                phase_name = row.get("phase_name", "??")
                run_number = row.get("run_number", "??")

                logger.info(
                    f"Loading scenario {index}: {scenario_id}-{phase_name}-{run_number}"
                )

                scenario_data = row["data"]

                scenario, completed_events = load_scenario(
                    scenario_data,
                    scenario_id,
                    load_completed_events,
                    hf_metadata=ExportedHuggingFaceMetadata(
                        dataset=dataset_name,
                        split=dataset_split,
                        revision=dataset_revision,
                    ),
                )
                if scenario is None or completed_events is None:
                    continue

                # Store HF source info for checkpoint metadata
                scenario._checkpoint_source_path = f"hf://{dataset_name}/{dataset_config}/{dataset_split}/{scenario_id}"

                # Set the run_number from the HuggingFace row metadata
                # This ensures the run_number is preserved from the dataset
                scenario.run_number = row.get("run_number")

                yield scenario, completed_events

        return scenario_generator(), total_count
    except Exception as e:
        logger.exception(
            f"Failed to load dataset {dataset_name}/{dataset_config}/{dataset_split}: {str(e)}"
        )
        return iter([]), 0


def huggingface_scenario_iterator(
    dataset_name: str,
    dataset_config: str,
    dataset_split: str = "validation",
    dataset_revision: str | None = None,
    load_completed_events: bool = False,
    limit: int | None = None,
    **kwargs,
) -> CountableIterator[tuple[BenchmarkScenarioImportedFromJson, list[CompletedEvent]]]:
    """Iterate over the scenarios in a HuggingFace dataset.

    This function loads scenarios from a HuggingFace dataset and yields them one by one.

    :param dataset_name: The name of the HuggingFace dataset
    :param dataset_config: The config (subset) of the dataset (e.g., "ambiguity", "execution", "time", "execution_a2a", "execution_noise")
    :param dataset_split: The split of the dataset (e.g., "test", "validation", "train"), defaults to "test"
    :param dataset_revision: Optional revision of the dataset
    :param load_completed_events: Whether to load completed events for offline validation
    :param limit: Maximum number of scenarios to load, if None loads all scenarios
    :return: CountableIterator of tuples containing (scenario, completed_events)
    :rtype: CountableIterator[tuple[BenchmarkScenarioImportedFromJson, list[CompletedEvent]]]
    """
    iterator, total_count = _create_huggingface_scenario_iterator(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        dataset_revision=dataset_revision,
        load_completed_events=load_completed_events,
        limit=limit,
        **kwargs,
    )

    return CountableIterator(iterator, total_count)
