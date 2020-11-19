from pathlib import Path
import dataclasses

PROJECT_ROOT = Path(__file__).parents[2]
RESULT_ROOT = PROJECT_ROOT / "results"


@dataclasses.dataclass
class AugmentParameters:
    min_scale: float = 1.0
    max_scale: float = 1.0
    shift_range: int = 8


@dataclasses.dataclass
class FlawlessParameters:
    classes: tuple = (0, 5)
    ratios: tuple = (0.6, 0.6)
    pickle_dir: str = None
    num_files_per_batch: int = 5
    # threshold parameters used in checking if a cropped image is edge or background.
    max_thresh: float = 1500
    variance_thresh: float = 7500
    # number of images created from a single cropped image by augmentation
    num_augment: int = 1


@dataclasses.dataclass
class FlawParameters:
    classes: tuple = (1, 2, 3, 4)
    ratios: tuple = (0.2, 1.0, 1.0, 1.0)


@dataclasses.dataclass
class DatasetParameters:
    batch_size: int = 100
    output_image_size: int = 24
    num_channel: int = 3
    num_process: int = 6
    augment_params: AugmentParameters = AugmentParameters()
    flawless_params: FlawlessParameters = FlawlessParameters()
    flaw_params: FlawParameters = FlawParameters()
