"""Explicit model/run profiles; changing a recipe changes its identity."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Recipe(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    name: str = Field(min_length=1, max_length=100)
    model_kind: Literal["auto", "qwen3_5_text"] = "auto"
    device: Literal["cpu", "cuda"] = "cpu"
    precision: Literal["float32", "float16", "bfloat16"] = "float32"
    quantization: Literal["none", "nf4"] = "none"
    rank: int = Field(default=16, ge=1, le=256)
    alpha: int = Field(default=32, ge=1, le=1024)
    rank_stabilized: bool = False
    enable_thinking: bool = False
    targets: list[str] = Field(default_factory=lambda: ["all-linear"], min_length=1)
    dropout: float = Field(default=0.05, ge=0, lt=1)
    learning_rate: float = Field(default=0.0001, gt=0, le=0.1)
    weight_decay: float = Field(default=0.01, ge=0, le=1)
    max_length: int = Field(default=2048, ge=16, le=32768)
    micro_batch: int = Field(default=1, ge=1, le=64)
    accumulation: int = Field(default=4, ge=1, le=128)
    steps: int = Field(default=100, ge=1, le=100000)
    warmup_steps: int = Field(default=10, ge=0)
    eval_every: int = Field(default=10, ge=1)
    max_grad_norm: float = Field(default=1.0, gt=0, le=100)
    max_amp_overflow_retries: int = Field(default=8, ge=0, le=32)
    max_run_seconds: int = Field(default=3600, ge=1, le=86400)
    gradient_checkpointing: bool = True
    seed: int = Field(default=92811, ge=0, le=2**31 - 1)
    cpu_threads: int = Field(default=2, ge=1, le=32)
    minimum_free_gpu_mib: int = Field(default=1024, ge=64)

    @model_validator(mode="after")
    def compatible(self):
        if self.warmup_steps >= self.steps:
            raise ValueError("Warmup must leave optimization steps")
        if self.device == "cpu" and (self.precision != "float32" or self.quantization != "none"):
            raise ValueError("CPU verification uses float32 and unquantized adapters")
        if self.quantization == "nf4" and self.precision == "float32":
            raise ValueError("NF4 recipe requires an explicitly selected mixed precision")
        if "all-linear" in self.targets and self.targets != ["all-linear"]:
            raise ValueError("Use all-linear or explicit module names")
        return self
