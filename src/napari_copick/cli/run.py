import sys
from typing import List

import click
import napari
from copick.cli.util import add_config_option, add_debug_option
from copick.util.log import get_logger

from napari_copick.widget import CopickPlugin


def run_napari(
    config: str = None,
    dataset_ids: list[int] = None,
    overlay_root: str = "/tmp/overlay_root",
) -> None:
    viewer = napari.Viewer()
    copick_plugin = CopickPlugin(
        viewer,
        config_path=config,
        dataset_ids=dataset_ids,
        overlay_root=overlay_root,
    )
    viewer.window.add_dock_widget(copick_plugin, area="right")
    napari.run()


@click.group()
@click.pass_context
def cli(ctx):
    pass


@cli.command(
    context_settings={"show_default": True},
    short_help="Start Napari with Copick Plugin.",
)
@add_config_option
@click.option(
    "-ds",
    "--dataset-ids",
    type=int,
    multiple=True,
    help="Dataset IDs to include in the project.",
    metavar="ID",
    default=(),
)
@click.option(
    "--overlay-root",
    type=str,
    default="/tmp/overlay_root",
    help="Root URL for the overlay storage when using dataset IDs.",
)
@add_debug_option
@click.pass_context
def run(
    ctx,
    config: str = None,
    dataset_ids: List[int] = None,
    overlay_root: str = "/tmp/overlay_root",
    debug: bool = False,
) -> None:
    """Open Napari with the copick plugin initialized."""
    logger = get_logger(__name__, debug)

    if not config and not dataset_ids:
        logger.critical("Either --config_path or --dataset_ids must be provided")
        sys.exit(1)
    elif config and dataset_ids:
        logger.critical("Only one of --config_path or --dataset_ids should be provided, not both")
        sys.exit(1)

    run_napari(config, dataset_ids, overlay_root=overlay_root)
