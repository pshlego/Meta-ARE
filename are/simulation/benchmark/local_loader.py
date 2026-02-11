# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import concurrent.futures
import logging
from typing import Iterator

from tqdm import tqdm

from are.simulation.scenarios.scenario_imported_from_json.benchmark_scenario import (
    BenchmarkScenarioImportedFromJson,
)
from are.simulation.types import CompletedEvent
from are.simulation.utils.countable_iterator import CountableIterator

logger: logging.Logger = logging.getLogger(__name__)


def _create_local_scenario_iterator(
    dataset_path: str,
    dataset_config: str | None = None,
    dataset_split: str | None = None,
    load_completed_events: bool = False,
    limit: int | None = None,
) -> tuple[
    Iterator[tuple[BenchmarkScenarioImportedFromJson, list[CompletedEvent]]], int
]:
    """Create an iterator over the scenarios in a local dataset directory.

    This function loads scenarios from a local directory or JSONL file and returns an iterator.
    It supports loading from nested directory structures based on config and split parameters.

    :param dataset_path: The path to the dataset directory or JSONL file
    :param dataset_config: The name of a subdirectory containing the config we want (if None, iterates over files in the current directory)
    :param dataset_split: The name of a subdirectory containing the split we want (if None, iterates over files in the current directory)
    :param load_completed_events: Whether to load completed events for offline validation
    :param limit: Maximum number of scenarios to load (if None, loads all scenarios)
    :return: Tuple of (iterator of tuples containing (scenario, completed_events), total count)
    :rtype: tuple[Iterator[tuple[BenchmarkScenarioImportedFromJson, list[CompletedEvent]]], int]
    """
    from are.simulation.benchmark.scenario_loader import (
        find_scenario_paths,
        load_scenario,
    )

    fs, scenario_paths = find_scenario_paths(
        dataset_path, dataset_config, dataset_split
    )

    # Get the total count of scenarios
    total_count = len(scenario_paths)
    logger.info(f"Found {total_count} scenarios in local dataset")

    # Apply limit if specified
    if limit is not None:
        logger.info(f"Limiting to {limit} scenarios from local dataset")
        scenario_paths = scenario_paths[:limit]
        total_count = min(total_count, limit)

    def load_local_scenario(
        scenario_path: str,
    ) -> tuple[BenchmarkScenarioImportedFromJson, list[CompletedEvent]] | None:
        try:
            with fs.open(scenario_path, "r") as f:
                scenario_str = f.read()
            scenario, completed_events = load_scenario(
                scenario_str,
                scenario_path,
                load_completed_events,
            )
            if scenario is None or completed_events is None:
                return None
            # Store the source file path for checkpoint metadata
            scenario._checkpoint_source_path = scenario_path
            logger.info(f"Successfully loaded scenario: {scenario_path}")
            return scenario, completed_events
        except FileNotFoundError:
            logger.error(f"Failed to load scenario {scenario_path}: File not found")
            return None
        except Exception as e:
            logger.error(f"Failed to load scenario {scenario_path}: {str(e)}")
            return None

    # Show progress bar for loading scenarios
    logger.info(f"Loading {len(scenario_paths)} scenarios from local dataset")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(load_local_scenario, path): path for path in scenario_paths
        }

        # Use tqdm to show progress of loading scenarios
        # Create a list to store the results
        results = []

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Loading scenarios",
            position=0,
            leave=True,
        ):
            result = future.result()
            if result is not None:
                results.append(result)

        # Return the iterator and total count
        return iter(results), total_count


def local_scenario_iterator(
    dataset_path: str,
    dataset_config: str | None = None,
    dataset_split: str | None = None,
    load_completed_events: bool = False,
    limit: int | None = None,
) -> CountableIterator[tuple[BenchmarkScenarioImportedFromJson, list[CompletedEvent]]]:
    """Iterate over the scenarios in a local dataset directory.

    This function loads scenarios from a local directory or JSONL file and yields them one by one.
    It supports loading from nested directory structures based on config and split parameters.

    :param dataset_path: The path to the dataset directory or JSONL file
    :param dataset_config: The name of a subdirectory containing the config we want (if None, iterates over files in the current directory)
    :param dataset_split: The name of a subdirectory containing the split we want (if None, iterates over files in the current directory)
    :param load_completed_events: Whether to load completed events for offline validation
    :param limit: Maximum number of scenarios to load (if None, loads all scenarios)
    :return: CountableIterator of tuples containing (scenario, completed_events)
    :rtype: CountableIterator[tuple[BenchmarkScenarioImportedFromJson, list[CompletedEvent]]]
    """
    iterator, total_count = _create_local_scenario_iterator(
        dataset_path=dataset_path,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        load_completed_events=load_completed_events,
        limit=limit,
    )

    return CountableIterator(iterator, total_count)
