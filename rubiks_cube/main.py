import ray
from gymnasium.wrappers import TimeLimit
from ray import air, tune
from ray.rllib.algorithms.appo import APPO, APPOConfig
from ray.tune.registry import register_env

from rubiks_cube.gym_env.cube_env import CubeEnv

select_env = "2x2x2_cube-v0"
register_env(select_env, lambda config: TimeLimit(CubeEnv(), max_episode_steps=50))

ray.init(num_gpus=1)

ray_config = APPOConfig()

ray_config.framework(framework="torch")

ray_config.resources(num_gpus=1, num_cpus_per_worker=0.5)

ray_config.rollouts(
    num_envs_per_worker=12,
    num_rollout_workers=4,
)


ray_config.environment(
    env=select_env,
    env_config={
        "max_episodes": 100,
    },
)

tuner = tune.Tuner(
    APPO,
    run_config=air.RunConfig(
        name="test-rubik",
        stop={"episode_reward_mean": 0.90},
        checkpoint_config=air.CheckpointConfig(checkpoint_frequency=5),
    ),
    param_space=ray_config.to_dict(),
)

results = tuner.fit()

print(results)
