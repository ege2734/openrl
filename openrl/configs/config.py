#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2023 The OpenRL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""""""
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from jsonargparse import ActionConfigFile, ArgumentParser

from openrl.configs.utils import ProcessYamlAction


def create_config_parser():
    """
    The configuration parser.
    """
    parser = ArgumentParser(
        description="openrl",
    )
    parser.add_argument("--config", action=ProcessYamlAction)
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    # For Transformers
    parser.add_argument("--encode_state", action="store_true", default=False)
    parser.add_argument("--n_block", type=int, default=1)
    parser.add_argument("--n_embd", type=int, default=64)
    parser.add_argument("--n_head", type=int, default=1)
    parser.add_argument("--dec_actor", action="store_true", default=False)
    parser.add_argument("--share_actor", action="store_true", default=False)

    parser.add_argument("--callbacks", type=List[dict])

    # For Stable-baselines3
    parser.add_argument(
        "--sb3_model_path",
        type=str,
        default=None,
        help="stable-baselines3 model path",
    )
    parser.add_argument(
        "--sb3_algo",
        type=str,
        default=None,
        help="stable-baselines3 algorithm",
    )

    # For Hierarchical RL
    parser.add_argument(
        "--step_difference",
        type=int,
        default=1,
        help="Frequency difference between Controller's step and Executor's step",
    )
    # For GAIL
    parser.add_argument(
        "--gail",
        action="store_true",
        default=False,
        help="do imitation learning with gail",
    )
    parser.add_argument(
        "--expert_data",
        type=str,
        default=None,
        help="directory that contains expert demonstrations for gail",
    )
    parser.add_argument(
        "--gail_batch_size",
        type=int,
        default=128,
        help="gail batch size (default: 128)",
    )
    parser.add_argument(
        "--dis_input_len", type=int, default=None, help="gail input length"
    )
    parser.add_argument(
        "--gail_loss_target",
        type=float,
        default=None,
        help="gail loss target at warm up",
    )
    parser.add_argument(
        "--gail_epoch", type=int, default=5, help="gail epochs (default: 5)"
    )
    parser.add_argument(
        "--gail_use_action",
        type=bool,
        default=True,
        help="whether to use action as the input of the gail discriminator",
    )
    parser.add_argument(
        "--gail_hidden_size",
        type=int,
        default=256,
        help="gail hidden state size (default: 256)",
    )
    parser.add_argument(
        "--gail_layer_num",
        type=int,
        default=3,
        help="gail hidden layer number (default: 3)",
    )
    parser.add_argument(
        "--gail_lr", type=float, default=5e-4, help="learning rate (default: 5e-4)"
    )
    # For Data Collector
    parser.add_argument(
        "--data_dir", type=str, default=None, help="data save directory."
    )
    parser.add_argument(
        "--force_rewrite",
        action="store_true",
        default=False,
        help="by default False, will delete the data save directory if it exists.",
    )
    parser.add_argument(
        "--collector_num", type=int, default=1, help="number of collectors"
    )
    # For convert
    parser.add_argument(
        "--input_data_dir", type=str, default=None, help="input save directory."
    )
    parser.add_argument(
        "--output_data_dir", type=str, default=None, help="output data directory."
    )
    parser.add_argument("--worker_num", type=int, default=1, help="number of workers")
    parser.add_argument(
        "--sample_interval", type=int, default=1, help="data sample interval"
    )
    # For Self-Play
    parser.add_argument(
        "--selfplay_api.host",
        default="127.0.0.1",
        type=str,
        help="host for selfplay api",
    )
    parser.add_argument(
        "--selfplay_api.port",
        default=10086,
        type=int,
        help="port for selfplay api",
    )
    parser.add_argument(
        "--lazy_load_opponent",
        default=True,
        type=bool,
        help=(
            "if true, when the opponents are the same opponent_type, will only load the"
            " weight. Otherwise, will load the pythoon script."
        ),
    )
    parser.add_argument(
        "--self_play",
        action="store_true",
        default=False,
        help="whether to use selfplay",
    )
    parser.add_argument(
        "--selfplay_algo",
        type=str,
        default="WeightExistEnemy",
        help="choose selfplay algorithm",
    )
    parser.add_argument(
        "--max_play_num",
        type=int,
        default=2000,
        help="upper bound of each enemy's play list length",
    )
    parser.add_argument(
        "--max_enemy_num",
        type=int,
        default=-1,
        help="upper bound of enemy model's number exclusive of existing enemies",
    )
    parser.add_argument(
        "--exist_enemy_num", type=int, default=0, help="exist enemy num"
    )
    parser.add_argument(
        "--random_pos",
        type=int,
        default=-1,
        help="random enemy model's position in enemy pool",
    )
    parser.add_argument(
        "--build_in_pos",
        type=int,
        default=-1,
        help="build-in enemy model's position in enemy pool",
    )
    # For AMP
    parser.add_argument(
        "--use_amp",
        type=bool,
        default=False,
        help="use mixed precision training",
    )
    # For Optimizer
    parser.add_argument(
        "--load_optimizer",
        action="store_true",
        default=False,
        help="whether to restore optimizer",
    )
    # For JRPO
    parser.add_argument(
        "--use_joint_action_loss",
        type=bool,
        default=False,
        help="whether to use joint action loss",
    )
    # For Game Wrapper
    parser.add_argument(
        "--frameskip",
        type=int,
        default=None,
        help="whether to use frameskip, default is None",
    )
    # For Evaluation
    parser.add_argument(
        "--eval_render",
        default=False,
        action="store_true",
        help="whether to render during evaluation",
    )
    # For JiDi evaluation
    # parser.add_argument("--switch_two_side", default=False, action="store_true",help="whether to evaluate twice to switch two side")
    # For Distributed Training
    parser.add_argument(
        "--terminal",
        default="current_terminal",
        choices=[
            "local",
            "current_terminal",
            "tmux_session",
            "ssh_tmux_session",
            "k8s",
            "k8s_single",
        ],
        help="which terminal to use",
    )
    parser.add_argument(
        "--distributed_type",
        type=str,
        default="sync",
        help="distributed type to use actors.",
        choices=["sync", "async"],
    )
    parser.add_argument(
        "--program_type",
        type=str,
        default="local",
        help="running type of current program.",
        choices=[
            "local",
            "whole",
            "actor",
            "learner",
            "server",
            "server_learner",
            "local_evaluator",
            "remote_evaluator",
        ],
    )
    parser.add_argument(
        "--share_temp_dir", default=None, help="temp directory to store job.pkl"
    )
    parser.add_argument(
        "--share_entry_script_path",
        default=None,
        help="common path for the process_entry.py file",
    )
    parser.add_argument("--learner_num", type=int, default=1, help="number of learners")
    parser.add_argument(
        "--fetch_num",
        type=int,
        default=1,
        help="number of actors' data to train for a learner",
    )
    parser.add_argument(
        "--tmux_prefix",
        default=None,
        type=str,
        help="prefix which will be added to tmux session",
    )
    parser.add_argument(
        "--kill_all",
        default=False,
        action="store_true",
        help="kill all the tmux session",
    )
    parser.add_argument(
        "--namespace", default="default", type=str, help="namespace of pods"
    )
    # For k8s
    parser.add_argument(
        "--mount_path", default=None, type=str, help="Volume mount path"
    )
    parser.add_argument(
        "--mount_name", default=None, type=str, help="Volume mount name"
    )
    parser.add_argument(
        "--persistent_volume_claim_name",
        default=None,
        type=str,
        help="Persistent volume claim name",
    )
    # For Debug
    parser.add_argument(
        "--disable_training",
        action="store_true",
        default=False,
        help="disable training",
    )
    # For Actor
    parser.add_argument(
        "--use_half_actor",
        action="store_true",
        default=False,
        help="whether to use half float for actors",
    )
    parser.add_argument(
        "--algorithm_name",
        type=str,
        default="ppo",
        help="The algorithm name.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="",
        help="an identifier to distinguish different experiment.",
    )

    parser.add_argument(
        "--gpu_usage_type",
        type=str,
        default="auto",
        choices=["auto", "single"],
        help=(
            "by default auto, will determine the GPU automatically. If using single,"
            " use only use single GPU."
        ),
    )
    parser.add_argument(
        "--disable_cuda",
        action="store_true",
        default=False,
        help="by default False, will use GPU to train; or else will use CPU;",
    )
    parser.add_argument(
        "--cuda_deterministic",
        action="store_false",
        default=True,
        help=(
            "by default, make sure random seed effective. if set, bypass such function."
        ),
    )
    parser.add_argument(
        "--pytorch_threads",
        type=int,
        default=1,
        help="Number of torch threads for training",
    )
    parser.add_argument(
        "--n_rollout_threads",
        type=int,
        default=32,
        help="Number of parallel envs for training rollout",
    )
    parser.add_argument(
        "--n_eval_rollout_threads",
        type=int,
        default=1,
        help="Number of parallel envs for evaluating rollout",
    )
    parser.add_argument(
        "--n_render_rollout_threads",
        type=int,
        default=1,
        help="Number of parallel envs for rendering rollout",
    )
    parser.add_argument(
        "--num_env_steps",
        type=int,
        default=int(10e6),
        help="Number of environment steps to train (default: 10e6)",
    )
    parser.add_argument(
        "--user_name",
        type=str,
        default="openrl",
        help="user name for the running process",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help=(
            "[for wandb usage], to specify entity for simply collecting training data."
        ),
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        default=False,
        help=(
            "[for wandb usage], by default False, will log date to wandb server. or"
            " else will use tensorboard to log data."
        ),
    )
    # env parameters
    parser.add_argument(
        "--env_name",
        type=str,
        default="StarCraft2",
        help="specify the name of environment",
    )
    parser.add_argument(
        "--scenario_name",
        type=str,
        default="default",
        help="specify the name of scenario",
    )
    parser.add_argument("--num_agents", type=int, default=1, help="number of players")
    parser.add_argument("--num_enemies", type=int, default=1, help="number of enemies")
    parser.add_argument(
        "--use_obs_instead_of_state",
        action="store_true",
        default=False,
        help="Whether to use global state or concatenated obs",
    )
    # replay buffer parameters
    parser.add_argument(
        "--episode_length", type=int, default=200, help="episode length for training"
    )
    parser.add_argument(
        "--eval_episode_length",
        type=int,
        default=200,
        help="episode length for evaluation",
    )
    parser.add_argument(
        "--max_episode_length",
        type=int,
        default=None,
        help="Max length for any episode",
    )
    # network parameters
    parser.add_argument(
        "--separate_policy",
        action="store_true",
        default=False,
        help="Whether agent separate the policy",
    )
    parser.add_argument(
        "--use_conv1d", action="store_true", default=False, help="Whether to use conv1d"
    )
    parser.add_argument(
        "--stacked_frames",
        type=int,
        default=1,
        help="Dimension of hidden layers for actor/critic networks",
    )
    parser.add_argument(
        "--use_stacked_frames",
        action="store_true",
        default=False,
        help="Whether to use stacked_frames",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="Dimension of hidden layers for actor/critic networks",
    )  # different network may need different size
    parser.add_argument(
        "--layer_N",
        type=int,
        default=1,
        help="Number of layers for actor/critic networks",
    )
    parser.add_argument(
        "--activation_id",
        type=int,
        default=1,
        help="choose 0 to use tanh, 1 to use relu, 2 to use leaky relu, 3 to use selu",
    )
    parser.add_argument(
        "--use_popart",
        default=False,
        type=bool,
        help="by default False, use PopArt to normalize rewards.",
    )
    parser.add_argument(
        "--dual_clip_ppo",
        default=False,
        type=bool,
        help="by default False, use dual-clip ppo.",
    )
    parser.add_argument(
        "--dual_clip_coeff",
        type=float,
        default=3,
        help="by default 3, use PopArt to normalize rewards.",
    )
    parser.add_argument(
        "--use_valuenorm",
        type=bool,
        default=True,
        help="by default False, use running mean and std to normalize rewards.",
    )
    parser.add_argument(
        "--use_feature_normalization",
        type=bool,
        default=False,
        help="Whether to apply layernorm to the inputs",
    )
    parser.add_argument(
        "--use_orthogonal",
        action="store_false",
        default=True,
        help=(
            "Whether to use Orthogonal initialization for weights and 0 initialization"
            " for biases"
        ),
    )
    parser.add_argument(
        "--gain", type=float, default=0.01, help="The gain # of last action layer"
    )
    parser.add_argument(
        "--cnn_layers_params",
        type=str,
        default=None,
        help="The parameters of cnn layer",
    )
    parser.add_argument(
        "--use_maxpool2d",
        action="store_true",
        default=False,
        help="Whether to apply layernorm to the inputs",
    )
    parser.add_argument(
        "--rnn_type",
        type=str,
        default="gru",
        choices=["gru", "lstm"],
        help="rnn types: gru or lstm",
    )
    parser.add_argument("--rnn_num", type=int, default=1, help="rnn layer number")
    # recurrent parameters
    parser.add_argument(
        "--use_naive_recurrent_policy",
        type=bool,
        default=False,
        help="Whether to use a naive recurrent policy",
    )
    parser.add_argument(
        "--use_recurrent_policy",
        type=bool,
        default=False,
        help="use a recurrent policy",
    )
    parser.add_argument(
        "--recurrent_N", type=int, default=1, help="The number of recurrent layers."
    )
    parser.add_argument(
        "--data_chunk_length",
        type=int,
        default=2,
        help="Time length of chunks used to train a recurrent_policy",
    )
    parser.add_argument(
        "--use_influence_policy",
        action="store_true",
        default=False,
        help="use a recurrent policy",
    )
    parser.add_argument(
        "--influence_layer_N",
        type=int,
        default=1,
        help="Number of layers for actor/critic networks",
    )
    # attn parameters
    parser.add_argument(
        "--use_attn",
        action="store_true",
        default=False,
        help=" by default False, use attention tactics.",
    )
    parser.add_argument(
        "--attn_N", type=int, default=1, help="the number of attn layers, by default 1"
    )
    parser.add_argument(
        "--attn_size",
        type=int,
        default=64,
        help="by default, the hidden size of attn layer",
    )
    parser.add_argument(
        "--attn_heads", type=int, default=4, help="by default, the # of multiply heads"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="by default 0, the dropout ratio of attn layer.",
    )
    parser.add_argument(
        "--use_average_pool",
        type=bool,
        default=True,
        help="by default True, use average pooling for attn model.",
    )
    parser.add_argument(
        "--use_attn_internal",
        action="store_false",
        default=True,
        help="by default True, whether to strengthen own characteristics",
    )
    parser.add_argument(
        "--use_cat_self",
        action="store_false",
        default=True,
        help="by default True, whether to strengthen own characteristics",
    )
    # optimizer parameters
    parser.add_argument(
        "--lr", type=float, default=5e-4, help="learning rate (default: 5e-4)"
    )
    parser.add_argument(
        "--tau", type=float, default=0.995, help="soft update polyak (default: 0.995)"
    )
    parser.add_argument(
        "--critic_lr",
        type=float,
        default=5e-4,
        help="critic learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--opti_eps",
        type=float,
        default=1e-5,
        help="RMSprop optimizer epsilon (default: 1e-5)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0, help="weight decay (defaul: 0)"
    )
    # behavior cloning parameters
    parser.add_argument(
        "--bc_epoch",
        type=int,
        default=2,
        help="number of behavior cloning epochs (default: 15)",
    )

    # ppo parameters
    parser.add_argument(
        "--ppo_epoch", type=int, default=10, help="number of ppo epochs (default: 15)"
    )
    parser.add_argument(
        "--use_policy_vhead",
        action="store_true",
        default=False,
        help="by default, do not use policy vhead. if set, use policy vhead.",
    )
    parser.add_argument(
        "--use_clipped_value_loss",
        action="store_false",
        default=True,
        help="by default, clip loss value. If set, do not clip loss value.",
    )
    parser.add_argument(
        "--clip_param",
        type=float,
        default=0.2,
        help="ppo clip parameter (default: 0.2)",
    )
    parser.add_argument(
        "--num_mini_batch",
        type=int,
        default=1,
        help="number of batches for ppo (default: 1)",
    )
    parser.add_argument(
        "--mini_batch_size",
        type=int,
        default=None,
        help="batch size (default: 1)",
    )
    parser.add_argument(
        "--policy_value_loss_coef",
        type=float,
        default=0.5,
        help="policy value loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--entropy_coef",
        type=float,
        default=0.01,
        help="entropy term coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--value_loss_coef",
        type=float,
        default=0.5,
        help="value loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--use_max_grad_norm",
        action="store_false",
        default=True,
        help="by default, use max norm of gradients. If set, do not use.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=10,
        help="max norm of gradients (default: 10)",
    )
    parser.add_argument(
        "--use_gae",
        default=True,
        type=bool,
        help="use generalized advantage estimation",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor for rewards (default: 0.99)",
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=0.95,
        help="gae lambda parameter (default: 0.95)",
    )
    parser.add_argument(
        "--use_proper_time_limits",
        default=False,
        type=bool,
        help="compute returns taking into account time limits",
    )
    parser.add_argument(
        "--use_huber_loss",
        action="store_false",
        default=True,
        help="by default, use huber loss. If set, do not use huber loss.",
    )
    parser.add_argument(
        "--use_value_active_masks",
        type=bool,
        default=True,
        help="by default True, whether to mask useless data in value loss.",
    )
    parser.add_argument(
        "--use_policy_active_masks",
        action="store_false",
        default=True,
        help="by default True, whether to mask useless data in policy loss.",
    )
    parser.add_argument(
        "--huber_delta", type=float, default=10.0, help=" coefficience of huber loss."
    )
    parser.add_argument(
        "--use_adv_normalize",
        type=bool,
        default=False,
        help="whether to normalize advantage",
    )
    # ppg parameters
    parser.add_argument(
        "--aux_epoch",
        type=int,
        default=5,
        help="number of auxiliary epochs (default: 4)",
    )
    parser.add_argument(
        "--clone_coef",
        type=float,
        default=1.0,
        help="clone term coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--use_single_network",
        action="store_true",
        default=False,
        help="share base network between policy network and value network",
    )
    # run parameters
    parser.add_argument(
        "--use_linear_lr_decay",
        default=False,
        type=bool,
        help="use a linear schedule on the learning rate",
    )
    # save parameters
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1,
        help="time duration between contiunous twice models saving.",
    )
    parser.add_argument(
        "--only_eval",
        default=False,
        action="store_true",
        help="only execute evaluation, default False.",
    )
    # log parameters
    parser.add_argument(
        "--log_interval",
        type=int,
        default=5,
        help="time duration between contiunous twice log printing.",
    )
    parser.add_argument(
        "--log_each_episode",
        type=bool,
        default=True,
        help="Whether to log each episode number.",
    )
    parser.add_argument(
        "--use_rich_handler",
        type=bool,
        default=True,
        help="whether to use rich handler to print log.",
    )
    # eval parameters
    parser.add_argument(
        "--use_eval",
        action="store_true",
        default=False,
        help=(
            "by default, do not start evaluation. If set`, start evaluation alongside"
            " with training."
        ),
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=25,
        help="time duration between contiunous twice evaluation progress.",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=32,
        help="number of episodes of the evaluation.",
    )
    # render parameters
    parser.add_argument(
        "--save_gifs",
        action="store_true",
        default=False,
        help="by default, do not save render video. If set, save video.",
    )
    parser.add_argument(
        "--use_render",
        action="store_true",
        default=False,
        help=(
            "by default, do not render the env during training. If set, start render."
            " Note: something, the environment has internal render process which is not"
            " controlled by this hyperparam."
        ),
    )
    parser.add_argument(
        "--render_episodes",
        type=int,
        default=5,
        help="the number of episodes to render a given env",
    )
    parser.add_argument(
        "--ifi",
        type=float,
        default=0.1,
        help="the play interval of each rendered image in saved video.",
    )
    # pretrained parameters
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="by default None. set the path to pretrained model.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="by default None. set the path to save info.",
    )
    parser.add_argument(
        "--init_dir",
        type=str,
        default=None,
        help="use exist enemy to init model; if init_dir, then don't use model_dir",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="root dir to save curves, logs and models.",
    )
    # replay buffer parameters
    parser.add_argument(
        "--use_transmit",
        action="store_true",
        default=False,
        help=(
            "by default, do not use transmit. If set`, use transmit as the replay"
            " buffer."
        ),
    )
    # reverb server address
    parser.add_argument(
        "--server_address", type=str, default=None, help="Replay buffer server address."
    )
    parser.add_argument(
        "--use_tlaunch", action="store_true", default=False, help="whether use tlaunch."
    )
    parser.add_argument("--actor_num", type=int, default=1, help="number of actors")

    # replay buffer parameters
    parser.add_argument(
        "--use_reward_normalization",
        action="store_true",
        default=False,
        help="Whether to normalize rewards in replay buffer",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=5000,
        help="Max # of transitions that replay buffer can contain",
    )
    parser.add_argument(
        "--popart_update_interval_step",
        type=int,
        default=2,
        help="After how many train steps popart should be updated",
    )
    # prioritized experience replay
    parser.add_argument(
        "--use_per",
        action="store_true",
        default=False,
        help="Whether to use prioritized experience replay",
    )
    parser.add_argument(
        "--per_alpha",
        type=float,
        default=0.6,
        help="Alpha term for prioritized experience replay",
    )
    parser.add_argument(
        "--per_beta_start",
        type=float,
        default=0.4,
        help="Starting beta term for prioritized experience replay",
    )
    parser.add_argument(
        "--per_eps",
        type=float,
        default=1e-6,
        help="Eps term for prioritized experience replay",
    )
    parser.add_argument(
        "--per_nu",
        type=float,
        default=0.9,
        help="Weight of max TD error in formation of PER weights",
    )
    # off-policy
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of buffer transitions to train on at once",
    )
    parser.add_argument(
        "--actor_train_interval_step",
        type=int,
        default=2,
        help="After how many critic updates actor should be updated",
    )
    parser.add_argument(
        "--train_interval_episode",
        type=int,
        default=1,
        help="Number of env steps between updates to actor/critic",
    )
    parser.add_argument(
        "--train_interval",
        type=int,
        default=100,
        help="Number of episodes between updates to actor/critic",
    )
    parser.add_argument(
        "--use_same_critic_obs",
        action="store_false",
        default=True,
        help="whether all agents share the same centralized observation, in mpe",
    )
    parser.add_argument(
        "--use_global_all_local_state",
        action="store_true",
        default=False,
        help="Whether to use available actions, in smac",
    )
    parser.add_argument(
        "--prev_act_inp",
        action="store_true",
        default=False,
        help="Whether the actor input takes in previous actions as part of its input",
    )
    parser.add_argument(
        "--target_update",
        type=int,
        default=10,
        help=(
            "After how many evaluation network updates target network should be updated"
        ),
    )
    # for DDPG
    parser.add_argument(
        "--var",
        type=float,
        default=0.5,
        help="Control the exploration variance of the generated actions",
    )
    parser.add_argument(
        "--actor_lr",
        type=float,
        default=0.001,
        help="The learning rate of actor network",
    )
    # for SAC
    parser.add_argument(
        "--auto_alph",
        type=bool,
        default=False,
        help="whether to use automatic alpha tuning",
    )
    parser.add_argument(
        "--alpha_value",
        type=float,
        default=0.2,
        help="The value of alpha",
    )
    parser.add_argument(
        "--alpha_lr",
        type=float,
        default=2e-4,
        help="The learning rate of temperature alpha",
    )
    # update parameters
    parser.add_argument(
        "--use_soft_update",
        action="store_false",
        default=True,
        help="Whether to use soft update",
    )
    parser.add_argument(
        "--hard_update_interval_episode",
        type=int,
        default=200,
        help="After how many episodes the lagging target should be updated",
    )
    # exploration parameters
    parser.add_argument(
        "--num_random_episodes",
        type=int,
        default=5,
        help="Number of episodes to add to buffer with purely random actions",
    )
    parser.add_argument(
        "--epsilon_start",
        type=float,
        default=1.0,
        help="Starting value for epsilon, for eps-greedy exploration",
    )
    parser.add_argument(
        "--epsilon_finish",
        type=float,
        default=0.05,
        help="Ending value for epsilon, for eps-greedy exploration",
    )
    parser.add_argument(
        "--epsilon_anneal_time",
        type=int,
        default=5000,
        help="Number of episodes until epsilon reaches epsilon_finish",
    )
    # qmix parameters
    parser.add_argument(
        "--use_double_q",
        action="store_false",
        default=True,
        help="Whether to use double q learning",
    )
    parser.add_argument(
        "--hypernet_layers",
        type=int,
        default=2,
        help="Number of layers for hypernetworks. Must be either 1 or 2",
    )
    parser.add_argument(
        "--mixer_hidden_dim",
        type=int,
        default=32,
        help="Dimension of hidden layer of mixing network",
    )
    parser.add_argument(
        "--hypernet_hidden_dim",
        type=int,
        default=64,
        help=(
            "Dimension of hidden layer of hypernetwork (only applicable if"
            " hypernet_layers == 2"
        ),
    )
    # rmatd3 parameters
    parser.add_argument(
        "--target_action_noise_std",
        default=0.2,
        help="Target action smoothing noise for matd3",
    )

    parser.add_argument(
        "--data_path",
        default=None,
        type=str,
        help="the path of the training data",
    )
    parser.add_argument(
        "--env.args",
        default={},
        type=dict,
        help="the args of the env",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="the path of the model",
    )
    parser.add_argument(
        "--use_share_model",
        type=bool,
        default=False,
        help="use one class to implement policy and value networks",
    )

    # rewards class
    parser.add_argument(
        "--reward_class.id",
        default=None,
        type=str,
        help="the id of the reward class",
    )
    parser.add_argument(
        "--reward_class.args",
        default={},
        type=dict,
        help="the parameters of the reward class",
    )

    # vec info class
    parser.add_argument(
        "--vec_info_class.id",
        default=None,
        type=str,
        help="the id of the vec env's info class",
    )
    parser.add_argument(
        "--vec_info_class.args",
        default={},
        type=dict,
        help="the parameters of the vec info class",
    )

    # vec info class
    parser.add_argument(
        "--eval_metrics",
        nargs="+",
        type=dict,
        default=[],
        help="the id of the vec env's info class",
    )

    # selfplay parameters
    parser.add_argument(
        "--disable_update_enemy",
        default=False,
        type=bool,
        help="whether update enemy model",
    )
    parser.add_argument(
        "--least_win_rate",
        default=0.5,
        type=float,
        help="least_win_rate",
    )
    parser.add_argument(
        "--recent_list_max_len",
        default=100,
        type=int,
        help="max length of recent player list",
    )
    parser.add_argument(
        "--latest_weight",
        default=0.5,
        type=float,
        help="latest_weight",
    )
    parser.add_argument(
        "--newest_pos",
        default=1,
        type=int,
        help="newest_pos",
    )
    parser.add_argument(
        "--newest_weight",
        default=0.5,
        type=float,
        help="newest_weight",
    )
    parser.add_argument(
        "--use_deepspeed",
        default=False,
        type=bool,
        help="whether to use deepspeed",
    )
    parser.add_argument(
        "--local_rank",
        default=-1,
        type=int,
        help="local_rank",
    )
    parser.add_argument(
        "--use_offload",
        default=False,
        type=bool,
        help="whether to use offload (deepspeed)",
    )
    parser.add_argument(
        "--use_fp16",
        default=False,
        type=bool,
        help="whether to use fp16 (deepspeed)",
    )

    return parser


# Nested dataclasses for structured arguments


@dataclass
class SelfPlayAPI:
    """Configuration for the self-play API."""

    host: str = "127.0.0.1"
    port: int = 10086


@dataclass
class Env:
    """Configuration for the environment."""

    args: Dict = field(default_factory=dict)


@dataclass
class RewardClass:
    """Configuration for the reward class."""

    id: Optional[str] = None
    args: Dict = field(default_factory=dict)


@dataclass
class VecInfoClass:
    """Configuration for the vectorized environment's info class."""

    id: Optional[str] = None
    args: Dict = field(default_factory=dict)


# Main configuration dataclass


@dataclass
class Config:
    """
    Main configuration dataclass based on the provided script arguments.
    """

    # Top-level arguments
    config: Optional[str] = None
    seed: int = 0

    # For Transformers
    encode_state: bool = False
    n_block: int = 1
    n_embd: int = 64
    n_head: int = 1
    dec_actor: bool = False
    share_actor: bool = False

    callbacks: List[dict] = field(default_factory=list)

    # For Stable-baselines3
    sb3_model_path: Optional[str] = None
    sb3_algo: Optional[str] = None

    # For Hierarchical RL
    step_difference: int = 1

    # For GAIL
    gail: bool = False
    expert_data: Optional[str] = None
    gail_batch_size: int = 128
    dis_input_len: Optional[int] = None
    gail_loss_target: Optional[float] = None
    gail_epoch: int = 5
    gail_use_action: bool = True
    gail_hidden_size: int = 256
    gail_layer_num: int = 3
    gail_lr: float = 5e-4

    # For Data Collector
    data_dir: Optional[str] = None
    force_rewrite: bool = False
    collector_num: int = 1

    # For convert
    input_data_dir: Optional[str] = None
    output_data_dir: Optional[str] = None
    worker_num: int = 1
    sample_interval: int = 1

    # For Self-Play
    selfplay_api: SelfPlayAPI = field(default_factory=SelfPlayAPI)
    lazy_load_opponent: bool = True
    self_play: bool = False
    selfplay_algo: str = "WeightExistEnemy"
    max_play_num: int = 2000
    max_enemy_num: int = -1
    exist_enemy_num: int = 0
    random_pos: int = -1
    build_in_pos: int = -1

    # For AMP
    use_amp: bool = False

    # For Optimizer
    load_optimizer: bool = False

    # For JRPO
    use_joint_action_loss: bool = False

    # For Game Wrapper
    frameskip: Optional[int] = None

    # For Evaluation
    eval_render: bool = False

    # For Distributed Training
    terminal: str = "current_terminal"
    distributed_type: str = "sync"
    program_type: str = "local"
    share_temp_dir: Optional[str] = None
    share_entry_script_path: Optional[str] = None
    learner_num: int = 1
    fetch_num: int = 1
    tmux_prefix: Optional[str] = None
    kill_all: bool = False
    namespace: str = "default"

    # For k8s
    mount_path: Optional[str] = None
    mount_name: Optional[str] = None
    persistent_volume_claim_name: Optional[str] = None

    # For Debug
    disable_training: bool = False

    # For Actor
    use_half_actor: bool = False

    # Core parameters
    algorithm_name: str = "ppo"
    experiment_name: str = ""
    gpu_usage_type: str = "auto"
    disable_cuda: bool = False
    cuda_deterministic: bool = True
    pytorch_threads: int = 1
    n_rollout_threads: int = 32
    n_eval_rollout_threads: int = 1
    n_render_rollout_threads: int = 1
    num_env_steps: int = 10000000
    user_name: str = "openrl"

    # Wandb parameters
    wandb_entity: Optional[str] = None
    disable_wandb: bool = False

    # Env parameters
    env_name: str = "StarCraft2"
    scenario_name: str = "default"
    num_agents: int = 1
    num_enemies: int = 1
    use_obs_instead_of_state: bool = False

    # Replay buffer parameters
    episode_length: int = 200
    eval_episode_length: int = 200
    max_episode_length: Optional[int] = None

    # Network parameters
    separate_policy: bool = False
    use_conv1d: bool = False
    stacked_frames: int = 1
    use_stacked_frames: bool = False
    hidden_size: int = 64
    layer_N: int = 1
    activation_id: int = 1
    use_popart: bool = False
    dual_clip_ppo: bool = False
    dual_clip_coeff: float = 3.0
    use_valuenorm: bool = True
    use_feature_normalization: bool = False
    use_orthogonal: bool = True
    gain: float = 0.01
    cnn_layers_params: Optional[str] = None
    use_maxpool2d: bool = False
    rnn_type: str = "gru"
    rnn_num: int = 1

    # Recurrent parameters
    use_naive_recurrent_policy: bool = False
    use_recurrent_policy: bool = False
    recurrent_N: int = 1
    data_chunk_length: int = 2
    use_influence_policy: bool = False
    influence_layer_N: int = 1

    # Attention parameters
    use_attn: bool = False
    attn_N: int = 1
    attn_size: int = 64
    attn_heads: int = 4
    dropout: float = 0.0
    use_average_pool: bool = True
    use_attn_internal: bool = True
    use_cat_self: bool = True

    # Optimizer parameters
    lr: float = 5e-4
    tau: float = 0.995
    critic_lr: float = 5e-4
    opti_eps: float = 1e-5
    weight_decay: float = 0.0

    # Behavior cloning parameters
    bc_epoch: int = 2

    # PPO parameters
    ppo_epoch: int = 10
    use_policy_vhead: bool = False
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    num_mini_batch: int = 1
    mini_batch_size: Optional[int] = None
    policy_value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    use_max_grad_norm: bool = True
    max_grad_norm: float = 10.0
    use_gae: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    use_proper_time_limits: bool = False
    use_huber_loss: bool = True
    use_value_active_masks: bool = True
    use_policy_active_masks: bool = True
    huber_delta: float = 10.0
    use_adv_normalize: bool = False

    # PPG parameters
    aux_epoch: int = 5
    clone_coef: float = 1.0
    use_single_network: bool = False

    # Run parameters
    use_linear_lr_decay: bool = False

    # Save parameters
    save_interval: int = 1
    only_eval: bool = False

    # Log parameters
    log_interval: int = 5
    log_each_episode: bool = True
    use_rich_handler: bool = True

    # Eval parameters
    use_eval: bool = False
    eval_interval: int = 25
    eval_episodes: int = 32

    # Render parameters
    save_gifs: bool = False
    use_render: bool = False
    render_episodes: int = 5
    ifi: float = 0.1

    # Pretrained parameters
    model_dir: Optional[str] = None
    save_dir: Optional[str] = None
    init_dir: Optional[str] = None
    run_dir: Optional[str] = None

    # Replay buffer parameters
    use_transmit: bool = False
    server_address: Optional[str] = None
    use_tlaunch: bool = False
    actor_num: int = 1
    use_reward_normalization: bool = False
    buffer_size: int = 5000
    popart_update_interval_step: int = 2

    # Prioritized Experience Replay (PER)
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_eps: float = 1e-6
    per_nu: float = 0.9

    # Off-policy parameters
    batch_size: int = 32
    actor_train_interval_step: int = 2
    train_interval_episode: int = 1
    train_interval: int = 100
    use_same_critic_obs: bool = True
    use_global_all_local_state: bool = False
    prev_act_inp: bool = False
    target_update: int = 10

    # For DDPG
    var: float = 0.5
    actor_lr: float = 0.001

    # For SAC
    auto_alph: bool = False
    alpha_value: float = 0.2
    alpha_lr: float = 2e-4

    # Update parameters
    use_soft_update: bool = True
    hard_update_interval_episode: int = 200

    # Exploration parameters
    num_random_episodes: int = 5
    epsilon_start: float = 1.0
    epsilon_finish: float = 0.05
    epsilon_anneal_time: int = 5000

    # QMIX parameters
    use_double_q: bool = True
    hypernet_layers: int = 2
    mixer_hidden_dim: int = 32
    hypernet_hidden_dim: int = 64

    # RMATD3 parameters
    target_action_noise_std: float = 0.2

    # Data and model paths
    data_path: Optional[str] = None
    env: Env = field(default_factory=Env)
    model_path: Optional[str] = None
    use_share_model: bool = False

    # Reward class
    reward_class: RewardClass = field(default_factory=RewardClass)

    # Vec info class
    vec_info_class: VecInfoClass = field(default_factory=VecInfoClass)
    eval_metrics: List[dict] = field(default_factory=list)

    # Self-play parameters
    disable_update_enemy: bool = False
    least_win_rate: float = 0.5
    recent_list_max_len: int = 100
    latest_weight: float = 0.5
    newest_pos: int = 1
    newest_weight: float = 0.5

    # DeepSpeed parameters
    use_deepspeed: bool = False
    local_rank: int = -1
    use_offload: bool = False
    use_fp16: bool = False
