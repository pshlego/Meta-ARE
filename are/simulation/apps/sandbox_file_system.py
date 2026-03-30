# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import logging
import os
import random
import re
import shutil
import tempfile
import uuid
from typing import IO, Any, Callable, Iterable, cast

import fsspec
from fsspec import AbstractFileSystem
from fsspec.asyn import functools

from are.simulation.apps.app import App, Protocol
from are.simulation.apps.utils.fallback_file_system import FallbackFileSystem
from are.simulation.config import (
    ARE_SIMULATION_SANDBOX_PATH,
    DEMO_FS_DIR,
    DEMO_FS_PATH,
    FS_PATH,
)
from are.simulation.core.mdconvert import MarkdownConverter
from are.simulation.tool_utils import OperationType, app_tool
from are.simulation.types import EventRegisterer, event_registered

logger = logging.getLogger(__name__)


# we have 2 metaclass here, ABC and _Cached - this creates an issue we need to solve by creating a new metaclass that combines the two
class FinalMeta(type(AbstractFileSystem), type(App)):
    pass


# This is the type returned by info/ls, but is not allowed by type test
# class FileDetails(TypedDict):
#     name: str
#     size: int | None
#     type: str  # file or directory


class SandboxLocalFileSystem(App, AbstractFileSystem, metaclass=FinalMeta):
    """
    A sandboxed local file system that restricts operations to a temporary directory, providing an isolated
    and controlled environment for file and directory manipulations. This class offers functionality for
    creating, reading, updating, and deleting files and directories, as well as methods for saving and
    restoring file system states.

    The SandboxLocalFileSystem ensures that all operations are confined to a designated temporary directory,
    preventing access outside the sandbox. It maintains compatibility with standard file system operations
    and integrates support for event logging and tool registration for enhanced tracking.

    Key Features:
        - File and Directory Management: Create, open, list, move, and remove files and directories
        - Path Validation: Ensures operations remain within the designated sandbox directory
        - State Management: Save and load file system state for repeatability and restoration
        - Random Sampling: Retrieve random samples of files or folders within the sandbox
        - Permissions Management: Set permissions for files and directories
        - Event Logging: Integrated event tracking for operations

    Notes:
        - The file system is non-cacheable to ensure isolated instances across parallel executions
        - Temporary directory paths are converted to absolute paths if necessary
        - Directories can be populated programmatically using specified population methods
        - Operations attempting to access paths outside the sandbox will raise a PermissionError
        - File system state can be saved and reloaded for scenario-based testing or persistence
    """

    # We make sure to mark this FS as non cacheable otherwise when running multiple scenarios in parallel
    # They will use the same FS
    cachable = False

    def __init__(
        self,
        name: str | None = None,
        sandbox_dir: str | None = None,
        state_directory: str = DEMO_FS_PATH,
    ):
        super().__init__(name, sandbox_dir)
        # Create a temporary sub-directory.
        # Locate it inside the session's sandbox directory if that exists, otherwise use the default location.
        self.tmpdir = tempfile.mkdtemp(
            dir=(sandbox_dir or ARE_SIMULATION_SANDBOX_PATH),
            prefix="are_simulation_fs_sandbox_",
        )

        # Create the fallback filesystem wrapper around the local filesystem
        self.local_fs: FallbackFileSystem = FallbackFileSystem(
            fsspec.filesystem("file"),
            base_root=self.tmpdir,
        )

        self.state_directory = state_directory

        # Make sure the temp dir exists
        self.local_fs.makedirs(self.tmpdir, exist_ok=True)
        logger.debug(f"Created temporary LocalFileSystem directory: {self.tmpdir}")

    def __del__(self):
        if self.local_fs.exists(self.tmpdir):
            self.local_fs.rm(self.tmpdir, recursive=True)
            logger.debug(f"Removed temporary LocalFileSystem directory: {self.tmpdir}")

    def get_implemented_protocols(self) -> list[Protocol]:
        return [Protocol.FILE_SYSTEM]

    def reset(self):
        # The reset logic is combined with load_state
        super().reset()

    def _validate_path(self, path: str) -> str:
        # Ensure the path is within the tmpdir
        if path == "/":
            full_path = self.tmpdir
        elif path.startswith("/"):
            # Handle paths with leading slash as relative to sandbox root
            full_path = os.path.abspath(os.path.join(self.tmpdir, path.lstrip("/")))
        else:
            # home folder should be inside the sandbox
            path = re.sub(r"^~/?", "home/userhome/", path)
            path = re.sub(r"^~([^/]+)/?", "home/\1/", path)
            full_path = os.path.abspath(os.path.join(self.tmpdir, path))

        if not full_path.startswith(self.tmpdir):
            raise PermissionError(f"Operation not allowed outside sandbox: {full_path}")
        return full_path

    def get_state(self) -> dict[str, Any]:
        with EventRegisterer.disable():

            def build_tree(abs_path: str) -> dict[str, Any]:
                # For the root (tmpdir itself), use empty name
                if abs_path == self.tmpdir or abs_path == self.tmpdir.rstrip("/"):
                    name = ""
                else:
                    name = os.path.basename(abs_path)

                # Use direct os calls to avoid triggering fallback stats
                # loading from remote filesystems (e.g. HuggingFace).
                # get_state() only needs names and types, not sizes/modes.
                if os.path.isdir(abs_path):
                    children = []
                    try:
                        entries = sorted(os.listdir(abs_path))
                    except OSError:
                        entries = []
                    for entry in entries:
                        child_abs = os.path.join(abs_path, entry)
                        children.append(build_tree(child_abs))
                    return {"name": name, "type": "directory", "children": children}
                else:
                    return {"name": name, "type": "file"}

            # Only include the files structure, not the tmpdir
            return {"files": build_tree(self.tmpdir), "tmpdir": self.tmpdir}

    def load_state(self, state_dict: dict[str, Any]):
        with EventRegisterer.disable():
            # remove existing files
            self.local_fs.rm(self.tmpdir, recursive=True)
            self.local_fs.mkdir(self.tmpdir)

            # Set to track expected paths
            def restore_tree(node: dict[str, Any], parent_path: str) -> set[str]:
                node_name = node["name"]
                node_type = node["type"]
                node_path = os.path.join(parent_path, node_name)

                paths = set()
                if node_type == "directory":
                    if not self.exists(node_path):
                        self.mkdir(node_path)
                    if "children" in node:
                        for child in node["children"]:
                            paths.update(restore_tree(child, node_path))
                elif node_type == "file":
                    # For files, we just create an empty placeholder
                    # The actual content will be copied lazily when accessed
                    if not self.exists(node_path):
                        with self.open(node_path, "w") as _:
                            pass
                    paths = {node_path}

                return paths

            # Restore the tree structure
            expected_paths = set()
            if "files" in state_dict and "children" in state_dict["files"]:
                # the first node in the state_dict is the root node, so we ignore it's name, we use '/' instead
                for child in state_dict["files"]["children"]:
                    expected_paths.update(restore_tree(child, "/"))

            # We use the demo_filesystem to back the files stored in the state
            # Set the fallback root in our FallbackFileSystem wrapper
            self.local_fs.set_fallback_root(
                self.state_directory,
                expected_paths,
            )

    def _get_relative_path(self, item_name: str) -> str:
        if item_name.startswith(self.tmpdir):
            # Get the relative path from tmpdir
            rel_path = os.path.relpath(item_name, self.tmpdir)
            # If it's the root directory, use '/'
            if rel_path == ".":
                rel_path = "/"
            # Add a leading slash for absolute paths
            elif not rel_path.startswith("/"):
                rel_path = "/" + rel_path
            return rel_path
        return item_name

    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def open(
        self,
        path: str,
        mode: str = "rb",
        block_size: int | None = None,
        cache_options: dict | None = None,
        compression: str | None = None,
        **kwargs: dict[str, Any],
    ):
        """
        Open a file for reading or writing.
        :param path: path to the file to open
        :param mode: mode to open the file in
        :param   block_size: Size of blocks to read/write
        :param   cache_options: Cache options
        :param   compression: Compression format
        :param kwargs: additional arguments to pass to the file system
        :returns: file handle to the opened file

        :example:
            open("/path/to/file.txt", "r")
        """
        real_path = self._validate_path(path)

        return self.local_fs.open(
            real_path,
            mode=mode,
            block_size=block_size,
            cache_options=cache_options,
            compression=compression,
            **kwargs,
        )

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def cat(
        self,
        path: str,
        recursive: bool = False,
        on_error: str = "raise",
        **kwargs: dict[str, Any],
    ) -> str | bytes:
        """
        Read the contents of a file.
        :param path: path to the file to read
        :param recursive: If True, recursively read files in directories
        :param on_error: What to do on error ('raise' or 'omit')
        :param kwargs: additional arguments to pass to the file system
        :returns: contents of the file, read from with mode 'rb'

        :example:
            cat("/path/to/file.txt")
        """
        real_path = self._validate_path(path)

        return self.local_fs.cat(
            real_path,
            recursive=recursive,
            on_error=on_error,
            **kwargs,
        )

    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def mkdir(
        self, path: str, create_parents: bool = True, **kwargs: dict[str, Any]
    ) -> None:
        """
        Create a directory.
        :param path: path to the directory to create
        :param create_parents: if True, create parent directories if they do not exist
        :param kwargs: additional arguments to pass to the file system
        :returns: None

        :example:
            mkdir("/path/to/directory")
        """
        path = self._validate_path(path)
        self.local_fs.mkdir(path, create_parents=create_parents, **kwargs)

    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def makedirs(self, path: str, exist_ok: bool = False) -> None:
        """
        Create a directory and any parent directories if they do not exist.
        :param path: path to the directory to create
        :param exist_ok: if True, do not raise an exception if the directory already exists
        :returns: None

        :example:
            makedirs("/path/to/nested/directory", exist_ok=True)
        """
        path = self._validate_path(path)
        self.local_fs.makedirs(path, exist_ok=exist_ok)

    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def mv(
        self,
        path1: str,
        path2: str,
        recursive: bool = False,
        maxdepth: int | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """
        Move a file or directory.
        :param path1: path to the file or directory to move
        :param path2: path to the destination
        :param recursive: If True, move directories recursively
        :param maxdepth: Maximum depth to recurse when recursive=True
        :param kwargs: additional arguments to pass to the file system
        :returns: None

        :example:
            mv("/path/to/source.txt", "/path/to/destination.txt")
        """
        # Get the real paths
        real_path1 = self._validate_path(path1)
        real_path2 = self._validate_path(path2)

        # Use the fallback filesystem to handle the move
        self.local_fs.mv(
            real_path1,
            real_path2,
            recursive=recursive,
            maxdepth=maxdepth,
            **kwargs,
        )

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def ls(
        self, path: str = ".", detail: bool = False, **kwargs: dict[str, Any]
    ) -> list[str | dict[str, Any]]:  # FileDetails]:
        """
        List the contents of a directory.
        :param path: path to list, defaults to '.'
        :param detail: if True, return detailed information about each file
        :param kwargs: additional arguments to pass to the file system
        :returns: list of files and directories in the directory

        :example:
            ls("/path/to/directory", detail=True)
        """
        path = self._validate_path(path)
        results = self.local_fs.ls(path, detail=detail, **kwargs)

        # Strip the tmpdir prefix from the paths in the results
        new_res = []
        for item in results:
            if isinstance(item, str):
                new_res.append(self._get_relative_path(item))
            elif "name" in item:
                item["name"] = self._get_relative_path(item["name"])
                new_res.append(item)

        return new_res

    # this is already protected by validate path, but TODO we may need to ask extra protection
    @app_tool()
    @event_registered(operation_type=OperationType.WRITE)
    def rm(
        self, path: str, recursive: bool = False, maxdepth: int | None = None
    ) -> None:
        """
        Remove a file or directory.
        :param path: path to the file or directory to remove
        :param recursive: if True, remove the directory and all its contents
        :param maxdepth: Maximum depth to recurse when recursive=True
        :returns: None

        :example:
            rm("/path/to/file.txt", recursive=False)
        """
        real_path = self._validate_path(path)
        self.local_fs.rm(real_path, recursive=recursive, maxdepth=maxdepth)

    # this is already protected by validate path, but TODO we may need to ask extra protection
    def rmdir(self, path: str) -> None:
        path = self._validate_path(path)
        self.local_fs.rmdir(path)

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def exists(self, path: str, **kwargs: dict[str, Any]):
        """
        Check if a file or directory exists.
        :param path: path to the file or directory to check
        :param kwargs: additional arguments to pass to the file system
        :returns: True if the file or directory exists, False otherwise

        :example:
            exists("/path/to/file.txt")
        """
        path = self._validate_path(path)
        return self.local_fs.exists(path, **kwargs)

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def display(self, path: str) -> bool:
        """
        Displays the file at the given path in the UI, but it will not be added to the agent context.
        :param path: path to the file to display
        :returns: True if the file exists, False if the file does not exist

        :example:
            display("/path/to/file.txt")
        """
        path = self._validate_path(path)
        return self.local_fs.exists(path)

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def info(
        self, path: str, **kwargs: dict[str, Any]
    ) -> dict[str, Any]:  # FileDetails:
        """
        Get information about a file or directory.
        :param path: path to the file or directory to get information about
        :param kwargs: additional arguments to pass to the file system
        :returns: information about the file or directory

        :example:
            info("/path/to/file.txt")
        """
        real_path = self._validate_path(path)

        # Use the fallback filesystem to get info
        result = self.local_fs.info(real_path, **kwargs)

        # Hide the real path in the result
        if "name" in result:
            result["name"] = self._get_relative_path(result["name"])

        return result  # type: ignore

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def tree(self, path: str = ".") -> str:
        """
        Generate a string representing the directory and file structure in a tree format.
        :param path: Path to start the tree view from, defaults to the root of the sandbox
        :returns: A string representing the directory tree

        :example:
            tree("/path/to/directory")
        """
        with EventRegisterer.disable():

            def _build_tree(current_path, current_level, tree_str):
                # List items in the directory
                items = self.ls(current_path, detail=False)
                # Iterate through each item
                for i, item in enumerate(sorted(items)):  # type: ignore
                    # Display indentation and connector symbols
                    indent = "    " * current_level
                    connector = "└── " if i == len(items) - 1 else "├── "
                    name = os.path.basename(item)

                    tree_str += f"{indent}{connector}{name}\n"

                    # If item is a directory, recursively build its contents
                    if self.isdir(item):
                        tree_str = _build_tree(item, current_level + 1, tree_str)
                return tree_str

            tree_str = _build_tree(path, 0, f"{os.path.basename(path) or '/'}\n")

            # Build the tree string from the validated path, starting at level 0
            _build_tree(path, 0, f"{os.path.basename(path)}\n")
            return tree_str

    def populate(self, pop_method: Callable):
        pop_method(self)

    def get_file_paths_list(self) -> Iterable[str]:
        """
        This function returns a list of all files in the file system.
        :returns: list of file paths
        """
        with EventRegisterer.disable():
            # Walk through all directories and files in the temporary directory
            for root, _, files in self.local_fs.walk(self.tmpdir):
                for file in files:
                    # Construct the full path to the file
                    full_path = os.path.join(root, file)  # type: ignore
                    # Convert full path to a path relative to tmpdir and yield it
                    yield full_path

    def get_file_content(self, path: str) -> bytes:
        """
        Read the contents of a file.
        :param path: path to the file to read
        :returns: contents of the file
        """
        real_path = self._validate_path(path)
        return self.local_fs.cat(real_path)  # type: ignore

    def get_sample_files(self, k: int = 1, seed: int = 0) -> list[str]:
        """
        This function returns a list of randomly selected files.
        :param k: number of files to return
        :param seed: seed for random number generation
        :returns: list of file paths (relative to sandbox root)
        """
        random.seed(seed)
        file_list = list(self.get_file_paths_list())
        k = min(k, len(file_list))
        if k == 0:
            return []
        samples = random.sample(list(file_list), k)
        # Convert full paths to relative paths (removing tmpdir prefix)
        relative_samples = [self._get_relative_path(path) for path in samples]
        return relative_samples

    def get_folder_paths_list(self) -> Iterable[str]:
        """
        This function returns the list of all folders in the file system.
        :returns: list of folder paths
        """
        with EventRegisterer.disable():
            # Walk through all directories and files in the temporary directory
            for root, _, _ in self.local_fs.walk(self.tmpdir):
                # Yield the current directory
                assert isinstance(root, str)
                yield root

    def get_sample_folders(self, k: int = 1, seed: int = 0) -> list[str]:
        """
        This function returns a list of randomly selected folders.
        :param k: number of folders to return
        :param seed: seed for random number generation
        :returns: list of folder paths
        """
        random.seed(seed)
        folder_list = self.get_folder_paths_list()
        samples = random.sample(list(folder_list), k)
        return samples

    @event_registered(operation_type=OperationType.WRITE)
    def set_permissions(self, path, permission):
        path = self._validate_path(path)
        os.chmod(path, permission)

    def save_file_system_state(self, state_name: str):
        """
        This function saves the current state of the file system as is
        to the shared data storage.
        :param state_name: name under which the state will be saved.
        Reuse this state_name to restore the state.
        """
        assert state_name != DEMO_FS_DIR, "Cannot save state as demo state"

        target_path = os.path.join(FS_PATH, state_name)
        if os.path.exists(target_path):
            target_path = os.path.join(FS_PATH, uuid.uuid4().hex)
            logger.warning(
                f"State {state_name} already exists. Saving as {target_path}"
            )
        os.makedirs(target_path)

        # Copy contents of self.tmpdir to target_path
        for item in os.listdir(self.tmpdir):
            s = os.path.join(self.tmpdir, item)
            d = os.path.join(target_path, item)
            if os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)

    def load_file_system_state(self, state_name: str):
        """
        This function restores the file system to the state saved
        with the save_file_system_state function.
        Uses lazy loading to avoid copying all files immediately.
        :param state_name: name of the state to restore.
        """
        if state_name == DEMO_FS_DIR:
            # If the state is the demo state, use the demo file system
            # this might be set as a env variable, so let's use the loaded value from config.py
            source_path = DEMO_FS_PATH
        else:
            source_path = os.path.join(FS_PATH, state_name)
        if not self.local_fs.exists(source_path):
            raise Exception(f"State {state_name} does not exist.")

        # Remove all files in the file system
        self.local_fs.rm(self.tmpdir, recursive=True)
        self.local_fs.mkdir(self.tmpdir)

        # Use the FallbackFileSystem to create placeholders and handle lazy loading
        self.local_fs.set_fallback_root(source_path)

    def load_file_system_from_path(self, path: str):
        """
        This function restores the file system from a given path.
        :param path: path to the directory containing the file system.
        """
        # Remove all files in the file system
        self.local_fs.rm(self.tmpdir, recursive=True)
        self.local_fs.mkdir(self.tmpdir)

        # Restore the saved state to the file system
        self.local_fs.set_fallback_root(path)

    def load_directory_from_path(self, from_dir: str, to_dir: str):
        """
        This function restores the file system from a given path.
        :param path: path to the directory containing the file system.
        :param target_dir: target directory to load the files to.
        """
        # Ensure the target directory exists
        target_path = os.path.join(self.tmpdir, to_dir)
        os.makedirs(target_path, exist_ok=True)

        # Load the directory from the path
        for item in os.listdir(from_dir):
            s = os.path.join(from_dir, item)
            d = os.path.join(target_path, item)
            if os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)

    @functools.cached_property
    def _md_converter(self) -> MarkdownConverter:
        """Get or create a MarkdownConverter instance."""
        return MarkdownConverter()

    @app_tool()
    @event_registered(operation_type=OperationType.READ)
    def read_document(self, file_path: str, max_lines: int | None = 20) -> str:
        """
        Read and extract text content from various document types including PDFs, Word documents,
        Excel spreadsheets, PowerPoint presentations, HTML files, and more. Returns the content
        as structured text.
        Setting max_lines is highly recommended to avoid long documents and filling up your memory.

        :param file_path: The path to the document file to read
        :param max_lines: Maximum number of lines to return. If set to None/null, returns the entire document, default is 20.
        :returns: The document content as structured text

        :example:
            read_document("/Documents/report.pdf")
            read_document("/Documents/data.xlsx", max_lines=50)
        """
        # Validate the path is within the sandbox
        real_path = self._validate_path(file_path)

        # Get file extension for the converter
        _, ext = os.path.splitext(file_path)

        # Read the file content through fsspec without materializing to disk
        try:
            with self.local_fs.open(real_path, "rb") as file_handle:
                # Check if file has content
                file_handle.seek(0, 2)  # Seek to end
                file_size = file_handle.tell()
                if file_size == 0:
                    raise ValueError(f"File is empty: {file_path}")

                file_handle.seek(0)  # Reset to beginning
                # Convert using the new IO-based interface
                # Cast to IO[bytes] for type compatibility
                file_io = cast("IO[bytes]", file_handle)
                result = self._md_converter.convert_io(file_io, file_extension=ext)
        except Exception as e:
            # If we can't read the file, it doesn't exist or there's an error
            raise FileNotFoundError(
                f"File not found or could not be read: {file_path}. Error: {str(e)}"
            )

        # Extract the text content
        content = result.text_content

        # Add title if available
        if hasattr(result, "title") and result.title:
            content = f"# {result.title}\n\n{content}"

        # Truncate to max_lines if specified
        if max_lines is not None and max_lines > 0:
            lines = content.split("\n")
            if len(lines) > max_lines:
                truncated_content = "\n".join(lines[:max_lines])
                truncated_content += f"\n\n[Document truncated. Showing {max_lines} of {len(lines)} lines]"
                return truncated_content

        return content


class Files(SandboxLocalFileSystem):
    """
    A sandboxed local file system that restricts operations to a temporary directory, providing an isolated
    and controlled environment for file and directory manipulations. This class offers functionality for
    creating, reading, updating, and deleting files and directories, as well as methods for saving and
    restoring file system states.

    The Files ensures that all operations are confined to a designated temporary directory,
    preventing access outside the sandbox. It maintains compatibility with standard file system operations
    and integrates support for event logging and tool registration for enhanced tracking.

    Key Features:
        - File and Directory Management: Create, open, list, move, and remove files and directories
        - Path Validation: Ensures operations remain within the designated sandbox directory
        - State Management: Save and load file system state for repeatability and restoration
        - Random Sampling: Retrieve random samples of files or folders within the sandbox
        - Permissions Management: Set permissions for files and directories
        - Event Logging: Integrated event tracking for operations

    Notes:
        - The file system is non-cacheable to ensure isolated instances across parallel executions
        - Temporary directory paths are converted to absolute paths if necessary
        - Directories can be populated programmatically using specified population methods
        - Operations attempting to access paths outside the sandbox will raise a PermissionError
        - File system state can be saved and reloaded for scenario-based testing or persistence
    """

    def __init__(self, name: str | None = None, sandbox_dir: str | None = None):
        super().__init__(name, sandbox_dir)
