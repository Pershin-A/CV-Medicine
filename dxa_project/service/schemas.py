from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

class Payload(BaseModel):
    model_config = ConfigDict(extra='forbid')

class Predict(Payload):
    paths: list[str] = Field(default_factory=list, max_length=10000)
    image_ids: list[str] = Field(default_factory=list, max_length=10000)
    model_version: str | None = None
    mode: Literal['single','batch'] = 'single'
    request_id: str | None = None

class Annotation(Payload):
    region: Literal['SPINE','LEG_LEFT','LEG_RIGHT']
    geometry: dict
    reviewed: list[Literal['spine','artifact','hip','hip_points','scoliosis','hip_mask','spine_crests']] = Field(default_factory=list)
    targets: dict[str,int | None] = Field(default_factory=dict)
    spacing_mm: tuple[float,float] | None = None
    expected_version: int = Field(default=0,ge=0)
    mask_result_id: str | None = None

class AugmentationConfig(Payload):
    positive_count: int = Field(default=5,ge=0,le=500)
    negative_count_by_target: dict[str,int] = Field(default_factory=dict)
    negative_source_count: int = Field(default=5,ge=0,le=500)
    seed: int = 42
    max_attempts_per_sample: int = Field(default=100,ge=1,le=1000)

class Enqueue(Payload):
    image_id: str
    annotation_version: int = Field(ge=1)
    config: AugmentationConfig = Field(default_factory=AugmentationConfig)
    request_id: str | None = None

class Register(Payload):
    checkpoints: str
    protocol: str
    epochs: dict[str,int] | None = None
    activate: bool = True

class TrainingSample(Payload):
    path: str | None = None
    image_id: str | None = None
    annotation_version: int | None = Field(default=None,ge=1)
    geometry_path: str | None = None
    geometry: dict | None = None
    region: Literal['SPINE','LEG_LEFT','LEG_RIGHT'] | None = None
    targets: dict[str,int | None] = Field(default_factory=dict)
    reviewed: list[str] = Field(default_factory=list)
    spacing_mm: tuple[float,float] | None = None

class TrainingDataset(Payload):
    images: list[TrainingSample] = Field(default_factory=list,max_length=10000)
    manifest_path: str | None = None

class FitAugmentation(Payload):
    n_pp: int = Field(default=5,ge=0,le=500)
    n_pn: int = Field(default=5,ge=0,le=500)
    n_nn: int = Field(default=5,ge=0,le=500)
    seed: int = 42
    max_attempts_per_sample: int = Field(default=100,ge=1,le=1000)

class PartialFit(Payload):
    base_model_version: str | None = None
    queue_ids: list[str] | None = None
    replay_per_task: int = Field(default=32,ge=1,le=10000)
    learning_rate: float = Field(default=2e-5,gt=0,le=.01)
    request_id: str | None = None
    human: TrainingDataset | None = None
    model: TrainingDataset | None = None
    augmentation: FitAugmentation = Field(default_factory=FitAugmentation)
