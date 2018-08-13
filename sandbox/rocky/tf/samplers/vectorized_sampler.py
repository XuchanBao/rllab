import pickle

import tensorflow as tf
from rllab.sampler.base import BaseSampler
from sandbox.rocky.tf.envs.parallel_vec_env_executor import ParallelVecEnvExecutor
from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor
from rllab.misc import tensor_utils
import numpy as np
from rllab.sampler.stateful_pool import ProgBarCounter
import rllab.misc.logger as logger
import itertools


class VectorizedSampler(BaseSampler):

    def __init__(self, algo, n_envs=None):
        super(VectorizedSampler, self).__init__(algo)
        self.n_envs = n_envs

    def start_worker(self, include_joint_coords=False):
        n_envs = self.n_envs
        if n_envs is None:
            n_envs = int(self.algo.batch_size / self.algo.max_path_length)
            n_envs = max(1, min(n_envs, 100))

        if getattr(self.algo.env, 'vectorized', False):
            self.vec_env = self.algo.env.vec_env_executor(
                n_envs=n_envs, max_path_length=self.algo.max_path_length,
                include_joint_coords=include_joint_coords)
        else:
            envs = [pickle.loads(pickle.dumps(self.algo.env)) for _ in range(n_envs)]
            self.vec_env = VecEnvExecutor(
                envs=envs,
                max_path_length=self.algo.max_path_length,
                include_joint_coords=include_joint_coords
            )
        self.env_spec = self.algo.env.spec

    def shutdown_worker(self):
        self.vec_env.terminate()

    def obtain_samples_for_visualization(self, include_joint_coords=False):
        tf_env = self.algo.env
        if hasattr(tf_env.wrapped_env, "stats_recorder"):
            setattr(tf_env.wrapped_env.stats_recorder, "done", None)

        import builtins
        builtins.visualize = True

        print("\nAbout to start video...")
        obs_dim = self.env_spec.observation_space.shape[0]
        obs = tf_env.reset()
        obs = self._add_joint_coords_to_obs(obs, include_joint_coords)
        horizon = 1000
        for horizon_num in range(1, horizon + 1):
            # action, _ = self.algo.policy.get_action(obs[:obs_dim])
            action, _ = self.algo.policy.get_action(obs)
            next_obs, reward, done, _info = tf_env.step(action, use_states=obs)
            obs = self._add_joint_coords_to_obs(next_obs, include_joint_coords)
            if done or horizon_num == horizon:
                break
        builtins.visualize = False

    def obtain_samples(self, itr, include_joint_coords=False):
        logger.log("Obtaining samples for iteration %d..." % itr)
        paths = []
        n_samples = 0

        obses = self.vec_env.reset()
        obses = self._add_joint_coords_to_obses(obses, include_joint_coords)
        obs_dim = self.env_spec.observation_space.shape[0]

        dones = np.asarray([True] * self.vec_env.num_envs)
        running_paths = [None] * self.vec_env.num_envs

        pbar = ProgBarCounter(self.algo.batch_size)
        policy_time = 0
        env_time = 0
        process_time = 0

        policy = self.algo.policy
        import time
        while n_samples < self.algo.batch_size:
            t = time.time()
            policy.reset(dones)
            actions, agent_infos = policy.get_actions(obses)
            policy_time += time.time() - t
            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions, use_states=obses)
            env_time += time.time() - t

            t = time.time()

            agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
            env_infos = tensor_utils.split_tensor_dict_list(env_infos)
            if env_infos is None:
                env_infos = [dict() for _ in range(self.vec_env.num_envs)]
            if agent_infos is None:
                agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
            for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                    rewards, env_infos, agent_infos,
                                                                                    dones):
                if running_paths[idx] is None:
                    running_paths[idx] = dict(
                        observations=[],
                        actions=[],
                        rewards=[],
                        env_infos=[],
                        agent_infos=[],
                    )
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)
                if done:
                    paths.append(dict(
                        observations=self.env_spec.observation_space.flatten_n(running_paths[idx]["observations"]),
                        actions=self.env_spec.action_space.flatten_n(running_paths[idx]["actions"]),
                        rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
                        env_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                    ))
                    n_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = None
            process_time += time.time() - t
            pbar.inc(len(obses))
            obses = self._add_joint_coords_to_obses(next_obses, include_joint_coords)

        pbar.stop()

        logger.record_tabular("PolicyExecTime", policy_time)
        logger.record_tabular("EnvExecTime", env_time)
        logger.record_tabular("ProcessExecTime", process_time)

        return paths, n_samples

    def _add_joint_coords_to_obses(self, obses, include_joint_coords):
        if include_joint_coords:
            try:
                inner_env = self._get_inner_env()
                if hasattr(inner_env, "env"):
                    inner_env = inner_env.env
                extended_obses = []
                for obs in obses:
                    extended_obses.append(self._add_joint_coords_to_obs(
                        obs, include_joint_coords, inner_env))
                return np.array(extended_obses)
            except AttributeError:
                inner_envs = self._get_inner_envs()
                extended_obses = []
                for obs_i in range(len(obses)):
                    extended_obses.append(self._add_joint_coords_to_obs(
                        obses[obs_i], include_joint_coords, inner_envs[obs_i]))
                return np.array(extended_obses)

        return obses

    def _add_joint_coords_to_obs(self, obs, include_joint_coords, inner_env=None):
        if include_joint_coords:
            if not inner_env:
                inner_env = self._get_inner_env()
                if hasattr(inner_env, "env"):
                    inner_env = inner_env.env
            return np.append(obs, inner_env.get_geom_xpos().flatten())
        return obs

    def _get_inner_env(self):
        env = self.vec_env.vec_env
        while hasattr(env, "env"):
            env = env.env
        if hasattr(env.wrapped_env, '_wrapped_env'):
            return env.wrapped_env._wrapped_env
        else:
            return env.wrapped_env.env.unwrapped

    def _get_inner_envs(self):
        inner_envs = []
        for env in self.vec_env.envs:
            while hasattr(env, "env"):
                env = env.env
            if hasattr(env.wrapped_env, '_wrapped_env'):
                inner_envs.append(env.wrapped_env._wrapped_env)
            else:
                inner_envs.append(env.wrapped_env.env.unwrapped)
        return inner_envs
