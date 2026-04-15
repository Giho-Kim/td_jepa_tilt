# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MUJOCO_GL"] = "egl"  # for headless rendering

import torch

torch.set_float32_matmul_precision("high")
torch._inductor.config.autotune_local_cache = False

import json
import time
import typing as tp
from pathlib import Path
from typing import Dict, List

import gymnasium
import numpy as np
import pydantic
import torch
import tyro
import wandb

from metamotivo.agents import Agent
from metamotivo.base import BaseConfig
from metamotivo.data_loading.dmc import DMCDataConfig
from metamotivo.data_loading.ogbench import OGBenchDataConfig
from metamotivo.envs.dmc import DMCEnvConfig
from metamotivo.envs.ogbench import OGBenchEnvConfig
from metamotivo.evaluations.dmc import DMCRewardEvalConfig
from metamotivo.evaluations.ogbench import OGBenchRewardEvalConfig
from metamotivo.misc.loggers import CSVLogger
from metamotivo.utils import EveryNStepsChecker, get_local_workdir, set_seed_everywhere

TRAIN_LOG_FILENAME = "train_log.txt"

CHECKPOINT_DIR_NAME = "checkpoint"


Env = DMCEnvConfig | OGBenchEnvConfig
DataLoading = DMCDataConfig | OGBenchDataConfig

# Stackoverflow #70914419
Evaluation = tp.Annotated[
    tp.Union[DMCRewardEvalConfig, OGBenchRewardEvalConfig],
    pydantic.Field(discriminator="name"),
]


class TrainConfig(BaseConfig):
    # The "pydantic.Field" field is used to explicitely tell which field is the discriminative
    # feature
    agent: Agent = pydantic.Field(discriminator="name")

    env: Env = pydantic.Field(discriminator="name")
    data: DataLoading = pydantic.Field(discriminator="name")

    relabel_dataset: bool = False

    work_dir: str = pydantic.Field(default_factory=lambda: get_local_workdir("train_dmc"))

    seed: int = 0
    log_every_updates: int = 10_000
    num_train_steps: int = 3_000_000
    checkpoint_every_steps: int = 50_000
    #250_000
    # WANDB
    use_wandb: bool = False
    wandb_ename: str | None = None
    wandb_gname: str | None = None
    wandb_pname: str | None = None

    # misc
    buffer_device: str | None = None  # if None, use the agent's device

    # eval
    # If you want to add more available evaluations, Update "Evaluations" type above
    evaluations: Dict[str, Evaluation] | List[Evaluation] = pydantic.Field(default_factory=lambda: [])

    eval_every_steps: int = 50_000

    tags: dict = pydantic.Field(default_factory=lambda: {})

    def model_post_init(self, context):
        if self.relabel_dataset:
            if not isinstance(self.env, (DMCEnvConfig, OGBenchEnvConfig)):
                raise ValueError("Relabeling is only supported for DMC and OGBench environments")

    def build(self):
        return Workspace(self)


def create_agent_or_load_checkpoint(work_dir: Path, cfg: TrainConfig, agent_build_kwargs: dict[str, tp.Any]):
    checkpoint_dir = work_dir / CHECKPOINT_DIR_NAME
    checkpoint_time = 0
    if checkpoint_dir.exists():
        # read train status
        with (checkpoint_dir / "train_status.json").open("r") as f:
            train_status = json.load(f)
        checkpoint_time = train_status["time"]

        print(f"Loading the agent at time {checkpoint_time}")
        agent = cfg.agent.object_class.load(checkpoint_dir, device=cfg.agent.model.device)
    else:
        agent = cfg.agent.build(**agent_build_kwargs)
    return agent, cfg, checkpoint_time


def init_wandb(cfg: TrainConfig
               ):

    exp_name = "dmc-offline-collect"
    wandb_name = exp_name
    wandb_config = cfg.model_dump()
    wandb.init(entity=cfg.wandb_ename, project=cfg.wandb_pname, group=cfg.wandb_gname, name=wandb_name, config=wandb_config, \
               dir="./_wandb")


class Workspace:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg

        sample_env, _ = cfg.env.build()

        self.obs_space = sample_env.observation_space
        assert isinstance(self.obs_space, gymnasium.spaces.Box), "Only Box observation spaces are supported"

        self.action_space = sample_env.action_space
        assert len(self.action_space.shape) == 1, "Only 1D action space is supported"
        self.action_dim = self.action_space.shape[0]

        print(f"Workdir: {self.cfg.work_dir}")
        self.work_dir = Path(self.cfg.work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)

        self.train_logger = CSVLogger(filename=self.work_dir / TRAIN_LOG_FILENAME)

        set_seed_everywhere(self.cfg.seed)

        self.agent, self.cfg, self._checkpoint_time = create_agent_or_load_checkpoint(
            self.work_dir,
            self.cfg,
            agent_build_kwargs=dict(obs_space=self.obs_space, action_dim=self.action_dim),
        )
        self.agent._model.train()

        if isinstance(self.cfg.evaluations, list):
            self.evaluations = {eval_cfg.name_in_logs: eval_cfg.build() for eval_cfg in self.cfg.evaluations}
        elif isinstance(self.cfg.evaluations, dict):
            self.evaluations = {name: eval_cfg.build() for name, eval_cfg in self.cfg.evaluations.items()}
        self.evaluate = len(self.evaluations) > 0
        self.eval_loggers = {name: CSVLogger(filename=self.work_dir / f"{name}.csv") for name, eval_cfg in self.evaluations.items()}

        if self.cfg.use_wandb:
            init_wandb(self.cfg)

        with (self.work_dir / "config.json").open("w") as f:
            f.write(self.cfg.model_dump_json(indent=4))

    def train(self):
        self.start_time = time.time()
        self.train_offline()

    def train_offline(self) -> None:
        buffer_device = self.agent.device if self.cfg.buffer_device is None else self.cfg.buffer_device
        relabel_fn = self.cfg.env.get_relabel_fn(self.cfg.env.task) if self.cfg.relabel_dataset else None
        replay_buffer, init_obs = self.cfg.data.build(buffer_device, self.cfg.agent.train.batch_size, self.cfg.env.frame_stack, relabel_fn)


        total_metrics = None
        fps_start_time = time.time()
        checkpoint_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.checkpoint_every_steps)
        eval_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.eval_every_steps)
        log_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.log_every_updates)

        for t in range(self._checkpoint_time, int(self.cfg.num_train_steps) + 1):
            if (t != self._checkpoint_time) and checkpoint_time_checker.check(t):
                checkpoint_time_checker.update_last_step(t)
                self.save(t, replay_buffer)

            if self.evaluate and eval_time_checker.check(t):
                eval_time_checker.update_last_step(t)
                self.eval(t, replay_buffer=replay_buffer)

            if t % 100 == 0:
                self.collect_online_data(
                    replay_buffer=replay_buffer,
                    num_episodes=100,
                    horizon=1000,
                    random_actions=False,
                )

            metrics = self.agent.update(replay_buffer, t, init_obs)

            # we need to copy tensors returned by a cudagraph module
            if total_metrics is None:
                total_metrics = {k: metrics[k].clone() for k in metrics.keys()}
            else:
                total_metrics = {k: total_metrics[k] + metrics[k] for k in metrics.keys()}

            if log_time_checker.check(t):
                # print(self.agent.z.mean().item(), self.agent.z.var().item())
                log_time_checker.update_last_step(t)
                m_dict = {}
                for k in sorted(list(total_metrics.keys())):
                    tmp = total_metrics[k] / (1 if t == 0 else self.cfg.log_every_updates)
                    m_dict[k] = np.round(tmp.mean().item(), 6)
                m_dict["duration"] = time.time() - self.start_time
                m_dict["FPS"] = (1 if t == 0 else self.cfg.log_every_updates) / (time.time() - fps_start_time)
                if self.cfg.use_wandb:
                    wandb.log(
                        {f"train/{k}": v for k, v in m_dict.items()},
                        step=t,
                    )
                print(m_dict)
                total_metrics = None
                fps_start_time = time.time()
        return

    def eval(self, t, replay_buffer):
        print(f"Starting evaluation at time {t}")
        evaluation_results = {}

        self.agent._model.train(False)

        # This will contain the results, mapping evaluation.cfg.name --> dict of metrics
        evaluation_results = {}
        for evaluation_name in self.evaluations:
            evaluation = self.evaluations[evaluation_name]
            logger = self.eval_loggers[evaluation_name]

            evaluation_metrics, wandb_dict = evaluation.run(
                timestep=t,
                agent_or_model=self.agent,
                replay_buffer=replay_buffer,
                logger=logger,
            )
            # For wandb dict, put it on wandb
            if self.cfg.use_wandb and wandb_dict is not None:
                wandb.log(
                    {f"eval/{evaluation_name}/{k}": v for k, v in wandb_dict.items()},
                    step=t,
                )

            evaluation_results[evaluation_name] = evaluation_metrics

        # ---------------------------------------------------------------
        self.agent._model.train()

        return evaluation_results

    def save(self, time: int, replay_buffer: Dict[str, tp.Any]) -> None:
        print(f"Checkpointing at time {time}")
        self.agent.save(str(self.work_dir / CHECKPOINT_DIR_NAME))
        with (self.work_dir / CHECKPOINT_DIR_NAME / "train_status.json").open("w+") as f:
            json.dump({"time": time}, f, indent=4)
    #######################################################################################################
    import numpy as np
    import torch

    def collect_online_data(
            self,
            replay_buffer,
            num_episodes: int,
            horizon: int,
            random_actions: bool = False,
    ):
        if not hasattr(self, "_online_env") or self._online_env is None:
            self._online_env = self._build_online_env()

        return self._collect_episodes(
            env=self._online_env,
            replay_buffer=replay_buffer,
            num_episodes=num_episodes,
            horizon=horizon,
            random_actions=random_actions,
        )


    def _build_online_env(self):
        env, _ = self.cfg.env.build()
        return env

    def _collect_episodes(
            self,
            env,
            replay_buffer,
            num_episodes: int,
            horizon: int,
            random_actions: bool = False,
    ):
        """
        Collect transitions episode-wise and append them to replay_buffer["train"].

        Args:
            num_episodes: number of episodes to collect
            horizon: max steps per episode
            random_actions: whether to sample random actions
        Returns:
            total number of collected transitions
        """
        assert num_episodes > 0
        assert horizon > 0

        obs_list = []
        action_list = []
        physics_list = []
        discount_list = []

        next_obs_list = []
        next_physics_list = []
        next_discount_list = []
        terminated_list = []

        total_steps = 0

        self.agent._model.train(False)
        with torch.no_grad():
            for _ in range(num_episodes):
                obs, info = env.reset()
                idx = torch.randint(0, self.agent.z.shape[0], (1,), device=self.agent.z.device)
                z_t = self.agent.z[idx]

                for _ in range(horizon):
                    if random_actions:
                        action = env.action_space.sample()
                    else:
                        obs_t = torch.as_tensor(
                            obs, device=self.agent.device, dtype=torch.float32
                        ).unsqueeze(0)


                        action = (
                            self.agent.act(obs_t, z_t, mean=False)
                            .squeeze(0)
                            .cpu()
                            .numpy()
                        )

                    next_obs, reward, terminated, truncated, next_info = env.step(action)
                    done = terminated or truncated

                    obs_list.append(np.asarray(obs, dtype=np.float32))
                    action_list.append(np.asarray(action, dtype=np.float32))
                    physics_list.append(np.asarray(info["physics"], dtype=np.float32))
                    discount_list.append(
                        np.array(
                            1.0 if info.get("discount", None) is None else info["discount"],
                            dtype=np.float32,
                        )
                    )

                    next_obs_list.append(np.asarray(next_obs, dtype=np.float32))
                    next_physics_list.append(
                        np.asarray(next_info["physics"], dtype=np.float32)
                    )
                    next_discount_list.append(
                        np.array(
                            1.0
                            if next_info.get("discount", None) is None
                            else next_info["discount"],
                            dtype=np.float32,
                        )
                    )
                    terminated_list.append(np.array(done, dtype=bool))

                    total_steps += 1

                    obs, info = next_obs, next_info

                    if done:
                        break

        self.agent._model.train()

        if total_steps == 0:
            return 0

        batch = {
            "observation": np.stack(obs_list, axis=0),
            "action": np.stack(action_list, axis=0),
            "physics": np.stack(physics_list, axis=0),
            # "discount": np.stack(discount_list, axis=0).reshape(-1, 1),
            "next": {
                "observation": np.stack(next_obs_list, axis=0),
                "physics": np.stack(next_physics_list, axis=0),
                # "discount": np.stack(next_discount_list, axis=0).reshape(-1, 1),
                "terminated": np.stack(terminated_list, axis=0).reshape(-1, 1),
            },
        }

        replay_buffer["train"].extend(batch)
        return total_steps
if __name__ == "__main__":
    # This is the bare minimum CLI interface to launch experiments, but ideally you should
    # launch your experiments from Python code (e.g., see under "scripts")
    workspace = tyro.cli(Workspace)
    # workspace.train()
    try:
        workspace.train()
    finally:
        if workspace.cfg.use_wandb:
            wandb.finish()
