# -*- coding: utf-8 -*-
"""
Created on 2023-07-11 (Tue) 15:32:10

Pipeline utilities

@author: I.Azuma
"""
import logging
import multiprocessing
import os
import sys
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import h5py
import pandas as pd
from tqdm.auto import tqdm


class PipelineStep(ABC):
    """Base pipelines step"""

    def __init__(
        self,
        save_path: Union[None, str, Path] = None,
        precompute: bool = True,
        link_path: Union[None, str, Path] = None,
        precompute_path: Union[None, str, Path] = None,
    ) -> None:
        """Abstract class that helps with saving and loading precomputed results

        Args:
            save_path (Union[None, str, Path], optional): Base path to save results to.
                When set to None, the results are not saved to disk. Defaults to None.
            precompute (bool, optional): Whether to perform the precomputation necessary
                for the step. Defaults to True.
            link_path (Union[None, str, Path], optional): Path to link the output directory
                to. When None, no link is created. Only supported when save_path is not None.
                Defaults to None.
            precompute_path (Union[None, str, Path], optional): Path to save the output of
                the precomputation to. If not specified it defaults to the output directory
                of the step when save_path is not None. Defaults to None.
        """
        assert (
            save_path is not None or link_path is None
        ), "link_path only supported when save_path is not None"

        name = self.__repr__()
        self.save_path = save_path
        if self.save_path is not None:
            self.output_dir = Path(self.save_path) / name
            self.output_key = "default_key"
            self._mkdir()
            if precompute_path is None:
                precompute_path = save_path

        if precompute:
            self.precompute(
                link_path=link_path,
                precompute_path=precompute_path)

    def __repr__(self) -> str:
        """Representation of a pipeline step.

        Returns:
            str: Representation of a pipeline step.
        """
        variables = ",".join(
            [f"{k}={v}" for k, v in sorted(self.__dict__.items())])
        return (
            f"{self.__class__.__name__}({variables})".replace(" ", "")
            .replace('"', "")
            .replace("'", "")
            .replace("..", "")
            .replace("/", "_")
        )

    def _mkdir(self) -> None:
        """Create path to output files"""
        assert (
            self.save_path is not None
        ), "Can only create directory if base_path was not None when constructing the object"
        if not self.output_dir.exists():
            self.output_dir.mkdir()

    def _link_to_path(self, link_directory: Union[None, str, Path]) -> None:
        """Links the output directory to the given directory.

        Args:
            link_directory (Union[None, str, Path]): Directory to link to
        """
        if link_directory is None or Path(
                link_directory).parent.resolve() == Path(self.output_dir):
            logging.info("Link to self skipped")
            return
        assert (
            self.save_path is not None
        ), f"Linking only supported when saving is enabled, i.e. when save_path is passed in the constructor."
        if os.path.islink(link_directory):
            if os.path.exists(link_directory):
                logging.info("Link already exists: overwriting...")
                os.remove(link_directory)
            else:
                logging.critical(
                    "Link exists, but points nowhere. Ignoring...")
                return
        elif os.path.exists(link_directory):
            os.remove(link_directory)
        os.symlink(self.output_dir, link_directory, target_is_directory=True)

    def precompute(
        self,
        link_path: Union[None, str, Path] = None,
        precompute_path: Union[None, str, Path] = None,
    ) -> None:
        """Precompute all necessary information for this step

        Args:
            link_path (Union[None, str, Path], optional): Path to link the output to. Defaults to None.
            precompute_path (Union[None, str, Path], optional): Path to load/save the precomputation outputs. Defaults to None.
        """
        pass

    def process(
        self, *args: Any, output_name: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Main process function of the step and outputs the result. Try to saves the output when output_name is passed.

        Args:
            output_name (Optional[str], optional): Unique identifier of the passed datapoint. Defaults to None.

        Returns:
            Any: Result of the pipeline step
        """
        if output_name is not None and self.save_path is not None:
            return self._process_and_save(
                *args, output_name=output_name, **kwargs)
        else:
            return self._process(*args, **kwargs)

    @abstractmethod
    def _process(self, *args: Any, **kwargs: Any) -> Any:
        """Abstract method that performs the computation of the pipeline step

        Returns:
            Any: Result of the pipeline step
        """

    def _get_outputs(self, input_file: h5py.File) -> Union[Any, Tuple]:
        """Extracts the step output from a given h5 file

        Args:
            input_file (h5py.File): File to load from

        Returns:
            Union[Any, Tuple]: Previously computed output of the step
        """
        outputs = list()
        nr_outputs = len(input_file.keys())

        # Legacy, remove at some point
        if nr_outputs == 1 and self.output_key in input_file.keys():
            return tuple([input_file[self.output_key][()]])

        for i in range(nr_outputs):
            outputs.append(input_file[f"{self.output_key}_{i}"][()])
        if len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)

    def _set_outputs(self, output_file: h5py.File,
                     outputs: Union[Tuple, Any]) -> None:
        """Save the step output to a given h5 file

        Args:
            output_file (h5py.File): File to write to
            outputs (Union[Tuple, Any]): Computed step output
        """
        if not isinstance(outputs, tuple):
            outputs = tuple([outputs])
        for i, output in enumerate(outputs):
            output_file.create_dataset(
                f"{self.output_key}_{i}",
                data=output,
                compression="gzip",
                compression_opts=9,
            )

    def _process_and_save(
        self, *args: Any, output_name: str, **kwargs: Any
    ) -> Any:
        """Process and save in the provided path as as .h5 file

        Args:
            output_name (str): Unique identifier of the the passed datapoint

        Raises:
            read_error (OSError): When the unable to read to self.output_dir/output_name.h5
            write_error (OSError): When the unable to write to self.output_dir/output_name.h5
        Returns:
            Any: Result of the pipeline step
        """
        assert (
            self.save_path is not None
        ), "Can only save intermediate output if base_path was not None when constructing the object"
        output_path = self.output_dir / f"{output_name}.h5"
        if output_path.exists():
            logging.info(
                f"{self.__class__.__name__}: Output of {output_name} already exists, using it instead of recomputing"
            )
            try:
                with h5py.File(output_path, "r") as input_file:
                    output = self._get_outputs(input_file=input_file)
            except OSError as read_error:
                print(f"\n\nCould not read from {output_path}!\n\n")
                raise read_error
        else:
            output = self._process(*args, **kwargs)
            try:
                with h5py.File(output_path, "w") as output_file:
                    self._set_outputs(output_file=output_file, outputs=output)
            except OSError as write_error:
                print(f"\n\nCould not write to {output_path}!\n\n")
                raise write_error
        return output