from mmengine.config import read_base

from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner, SlurmRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    # from .datasets.collections.lclm_long import datasets
    from .datasets.collections.long_score_long import datasets
    # from .mymodel.origin_scaling_rope_long import models
    # from .mymodel.origin_llama2 import models
    from .mymodel.scaling_rope_long import models

work_dir = './outputs/scaling_rope_long/'

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000, gen_task_coef=15),
    # partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=2,
        task=dict(type=OpenICLInferTask),
        retry=4),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=2,
        task=dict(type=OpenICLEvalTask),
        retry=4),
)

# python run.py configs/eval_scaling_rope_long.py -p llm_o --debug 调试用
# python run.py configs/eval_scaling_rope_long.py -p llm_o 第一次用
# python run.py configs/eval_scaling_rope_long.py -p llm_o -r 第二次用
