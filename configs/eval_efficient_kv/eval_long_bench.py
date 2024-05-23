from mmengine.config import read_base

from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import LocalRunner, SlurmRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from copy import deepcopy
with read_base():
    from ..datasets.collections.long_score_long import datasets
    from ..mymodel.compresser import template, all_models, paths, pruner_types, fuser_types, memory_types, chunk_sizes, d_querys, compressed_chunk_sizes

llm=["llama2_7B"]
fuser = [None]
pruner=['conv']
memory=["dynamic_incremental_double_compress"]
chunk_size=[512]
compressed_chunk_size=[c_s // 8 for c_s in chunk_size]
dq=[1024]
is_trained=[True]

# all_models存储了事先定义好的所有可能的设置，而models存储了在当下文件中需要测试的所有设 置
models = []
for l in llm:
  for p in pruner:
    for f in fuser:
      for m in memory:
        for c in chunk_size:
          for cc in compressed_chunk_size:
            for d in dq:
              for i in is_trained:                                
                try:
                    models.append(
                        all_models[template.format(
                                    llm=l, 
                                    pruner=p, 
                                    fuser=f, 
                                    memory=m, 
                                    chunk=c, 
                                    compressed_chunk=cc, 
                                    dq=d, 
                                    trained=i)]
                    )
                except Exception as e:
                    assert l in paths.keys()
                    assert p in pruner_types
                    assert f in fuser_types
                    assert m in memory_types
                    assert c in chunk_sizes
                    assert cc in compressed_chunk_sizes
                    assert d in d_querys
                    raise e
# models[0]['compresser_config']['perceiver_path'] = '/remote-home/zyzeng/collie/ckpts/llmllama2_7b#fuserllm#memoryChunk_Streaming#lr2e-05#chunk512#compress64/epoch_0-batch_1000/'
print(models)
del all_models # all_models contain too many configs, which can cause error

abbr = models[0]['abbr']
work_dir = f'./outputs/eval_long_bench/{abbr}'
infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=1000, gen_task_coef=15),
    # partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=4,
        task=dict(type=OpenICLInferTask),
        retry=1),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=2,
        task=dict(type=OpenICLEvalTask),
        retry=1),
)