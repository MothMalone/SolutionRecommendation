import gc
import math
import os
import time
from copy import deepcopy

import numpy as np
from loguru import logger

import comp
from config import AgentConfig, Config
from ctxpipe.agent.dqn import Agent
from ctxpipe.env.enviroment import Environment
from ctxpipe.env.primitives.imputercat import ImputerCatPrim
from ctxpipe.env.primitives.primitive import Primitive

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class Tester:

    def __init__(self, agent: Agent, env: Environment, test_pred, config: AgentConfig):
        self.agent = agent
        self.agent.no_random = True

        self.env = env
        self.test_pred = test_pred

        self._config = config
        self.epsilon_start = self._config.epsilon_start
        self.epsilon_final = self._config.epsilon_min
        self.epsilon_decay = self._config.eps_decay

        self.outputdir = self._config.model_dir

    def _epsilon_by_frame(self, frame_id):
        return self.epsilon_final + (
            self.epsilon_start - self.epsilon_final
        ) * math.exp(-1.0 * frame_id / self.epsilon_decay)

    def _choose_step_for_component(
        self,
        curr_component: str,
        state,
        tried_list: list,
        epsilon: float,
        has_num_nan: bool,
        has_cat_nan: bool,
    ):
        """Pick next candidate step for current component.

        This method is inference-safe: it always returns a valid (action, step) pair
        and provides blank fallbacks when model actions keep failing/repeating.
        """
        action = -1
        step = Primitive()

        if curr_component == "ImputerNum":
            if has_num_nan:
                action, _ = self.agent.act(
                    self.env.pipeline,
                    state,
                    curr_component,
                    tried_list,
                    epsilon,
                )
                action = int(action)
                action = min(max(action, 0), len(comp.imputernums) - 1)
                step = deepcopy(comp.imputernums[action])
            else:
                action = len(comp.imputernums)
                step = Primitive()

        elif curr_component == "ImputerCat":
            if has_cat_nan and -1 not in tried_list:
                action = -1
                step = ImputerCatPrim()
            else:
                # If imputercat failed once (or cat NaN is absent), force blank.
                action = -2
                step = Primitive()

        elif curr_component == "Encoder":
            if self.env.has_cat_cols():
                action, _ = self.agent.act(
                    self.env.pipeline,
                    state,
                    curr_component,
                    tried_list,
                    epsilon,
                )
                action = int(action)
                action = min(max(action, 0), len(comp.encoders) - 1)
                step = deepcopy(comp.encoders[action])
            else:
                action = len(comp.encoders)
                step = Primitive()

        elif curr_component == "FeaturePreprocessing":
            action, _ = self.agent.act(
                self.env.pipeline,
                state,
                curr_component,
                tried_list,
                epsilon,
            )
            action = int(action)
            action = min(max(action, 0), len(comp.fpreprocessings) - 1)
            step = deepcopy(comp.fpreprocessings[action])

        elif curr_component == "FeatureEngine":
            action, _ = self.agent.act(
                self.env.pipeline,
                state,
                curr_component,
                tried_list,
                epsilon,
            )
            action = int(action)
            action = min(max(action, 0), len(comp.fengines) - 1)
            step = deepcopy(comp.fengines[action])

        elif curr_component == "FeatureSelection":
            action, _ = self.agent.act(
                self.env.pipeline,
                state,
                curr_component,
                tried_list,
                epsilon,
            )
            action = int(action)
            action = min(max(action, 0), len(comp.fselections) - 1)
            step = deepcopy(comp.fselections[action])

        return action, step

    def get_five_items_from_pipeline(
        self,
        fr,
        state,
        reward_dic,
        seq,
        taskid,
        need_save=True,
        dataset_name="UNKNOWN",
        tag: str = "0",
    ):
        tryed_list = []
        epsilon = self._epsilon_by_frame(fr)
        pipeline_index = self.env.pipeline.get_index()
        has_num_nan, has_cat_nan = self.env.has_nan()

        logic_pipeline_id = self.env.pipeline.logic_pipeline_id
        curr_component = comp.lpipelines[logic_pipeline_id][pipeline_index]
        max_retry = int(os.getenv("CTXPIPE_INFER_MAX_RETRY", "24"))
        step_result = None
        err = None
        action = None

        for _ in range(max_retry):
            try:
                action, step = self._choose_step_for_component(
                    curr_component=curr_component,
                    state=state,
                    tried_list=tryed_list,
                    epsilon=epsilon,
                    has_num_nan=has_num_nan,
                    has_cat_nan=has_cat_nan,
                )
            except Exception as e:
                logger.warning(
                    f"[INFER] choose action failed for {curr_component}: {e}. fallback blank"
                )
                action, step = -999, Primitive()

            if action in tryed_list:
                continue

            tryed_list.append(action)
            try:
                step_result, err = self.env.step(step, has_timeout=False)
            except Exception as e:
                logger.warning(f"[INFER] env.step crashed on {curr_component}: {e}")
                step_result, err = None, -1

            if step_result is not None:
                break

        if step_result is None:
            # Last-resort hard fallback: blank should be no-op and must progress.
            logger.error(
                f"[INFER] component {curr_component} failed after {max_retry} retries; forcing blank"
            )
            blank_step = Primitive()
            try:
                step_result, err = self.env.step(blank_step, has_timeout=False)
                step = blank_step
            except Exception as e:
                logger.error(
                    f"[INFER] blank fallback crashed for {curr_component}: {e}"
                )
                step_result = None

        """get (st, r, st+1, done) for this execute"""
        if step_result is None:
            # Never crash inference loop on a bad operator; mark failure score.
            logger.error(
                f"[INFER] unrecoverable failure at component {curr_component}; assigning reward=-1"
            )
            reward = -1.0
            done = True
            seq.append("blank")
            try:
                self.end_time = time.time()
                self.env.reset(
                    taskid=taskid,
                    default=False,
                    metric=comp.metrics[0],
                    predictor=comp.predictors[self.test_pred],
                )
                self.env.pipeline.logic_pipeline_id, _ = self.agent.act(
                    self.env.pipeline,
                    self.env.lpip_state,
                    "LogicPipeline",
                    epsilon=self._epsilon_by_frame(0),
                )
                state = self.env.get_state()
            except Exception as e:
                logger.warning(f"[INFER] reset after failure failed: {e}")
            return state, reward_dic, seq, reward, done

        state, reward, next_state, done = step_result
        seq.append(step.name)
        state = next_state

        """if done, evaluate and save result"""
        if done:
            try:
                with open(self._config.pipelines_file_name, "a") as f:
                    f.write(
                        f"{tag}\t{dataset_name}\t{self.env.pipeline.sequence}\t{reward}\n"
                    )
            except Exception as e:
                logger.warning(f"[INFER] write pipelines.tsv failed: {e}")

            self.end_time = self.env.end_time
            self.env.reset(
                taskid=taskid,
                default=False,
                metric=comp.metrics[0],
                predictor=comp.predictors[self.test_pred],
            )
            self.env.pipeline.logic_pipeline_id, _ = self.agent.act(
                self.env.pipeline,
                self.env.lpip_state,
                "LogicPipeline",
                epsilon=self._epsilon_by_frame(0),
            )
            if self.env.pipeline.taskid not in reward_dic:
                reward_dic[self.env.pipeline.taskid] = {
                    "reward": {},
                    "seq": {},
                    "time": {},
                }

            reward_dic[self.env.pipeline.taskid]["reward"][self.pre_fr] = reward
            reward_dic[self.env.pipeline.taskid]["seq"][self.pre_fr] = seq
            reward_dic[self.env.pipeline.taskid]["time"][self.pre_fr] = (
                self.end_time - self.start_time
            )

            if need_save:
                np.save(self._config.test_reward_dic_file_name, reward_dic)

        return state, reward_dic, seq, reward, done

    def inference(
        self, data_path, tag: str = "56000", dataset_name="UNKNOWN"
    ):  # -> tuple[Any | list[Any], Any]:
        self.agent.load_weights(self.outputdir, tag=tag)
        self.pre_fr = 0

        score = 0
        reward_dic = {}
        datasetname = data_path.split("/")[-2]

        i = None
        for taskid in self._config.classification_task_dic:
            if datasetname == self._config.classification_task_dic[taskid]["dataset"]:
                i = taskid

        if i is None:
            logger.error(f"Invalid dataset mapping for path: {data_path}")
            return [], -1.0

        seq = []
        select_cl = 0
        for cl in comp.predictors:
            if cl.name == self._config.classification_task_dic[i]["model"]:
                select_cl = cl

        self.start_time = time.time()
        self.env.reset(
            taskid=i,
            default=False,
            metric=comp.metrics[0],
            predictor=select_cl,
        )
        self.env.pipeline.logic_pipeline_id, _ = self.agent.act(
            self.env.pipeline,
            self.env.lpip_state,
            "LogicPipeline",
            epsilon=self._epsilon_by_frame(0),
        )

        state = self.env.get_state()

        reward = None
        for fr in range(self.pre_fr + 1, self.pre_fr + 7):
            try:
                state, reward_dic, seq, reward, done = (
                    self.get_five_items_from_pipeline(
                        fr,
                        state,
                        reward_dic,
                        seq,
                        taskid=i,
                        need_save=False,
                        dataset_name=dataset_name,
                        tag=tag,
                    )
                )
            except Exception as e:
                logger.error(f"[INFER] fatal exception at frame {fr}: {e}")
                reward = -1.0
                break

        if reward is None:
            reward = -1.0

        score = reward

        return seq, score
