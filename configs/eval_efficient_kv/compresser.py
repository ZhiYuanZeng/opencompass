
from opencompass.models.collie_model_wrapper import PrunerModel

memory_types = [
  "Chunk_Streaming",
  "Incremental_Chunk_Streaming_Fixed_History",
  "Incremental_Chunk_Streaming_Dynamic_History",
  "dynamic_incremental_double_compress",
  None
]

pruner_types = [
    "h2o",
    "streaming_llm",
    "chunk_prefix",
    "tova",
    "random",
    "local_window",
    "no_compress",
    "perceiver",
    "conv",
    "roco",
    None
]

fuser_types = [
  'perceiver',
  'llm',
  None
]

paths = {
  'llama2_7B': '/remote-home/share/models/llama_v2_hf/7b', 
  'tiny_llama': '/remote-home/share/personal/zyzeng/models/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0',
  'internlm2_7b': '/remote-home/share/models/models--internlm2-7b',
  'internlm2_base_7b': '/remote-home/share/models/internlm2-7b-base-hf',
  'internlm_huawei7b': '/remote-home/share/models/mossHuawei/'
}

num_gpus = {
  'llama2_7B': 1, 
  'tiny_llama': 1,
  'internlm2_7b': 1,
  'internlm2_base_7b': 1,
  'internlm_huawei7b': 1
}

chunk_sizes = (64, 128, 256, 512, 1024, 2048, 4096)
compressed_chunk_sizes = (8, 16, 32, 64, 128, 256, 512)
d_querys = (128, 256, 512, 1024, 2048)
template = 'llm-{llm}#pruner-{pruner}#fuser-{fuser}#memory-{memory}#chunk-{chunk}#compressed_chunk-{compressed_chunk}#dq-{dq}#trained-{trained}'

def build_model_cfg(llm_name, chunk_size, d_query, compressed_chunk_size, pruner_type, fuser_type, memory_type, is_trained):
  pe_config = {
    'exp': False, '1d': False, 'imp': False, 'log': False, 
    'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
    'pi_lambda': 1, 'base': 10000.0, 'ntk_option': 'dynamic', 'ntk_alpha': 1., 
  }

  compresser_config = {
    "chunk_size": chunk_size,
    "d_query": d_query,
    "pruner_type": pruner_type,
    "fuser_type": fuser_type,
    "compressed_chunk_size": compressed_chunk_size,
    "memory_type": memory_type,
    "perceiver_path": None
  }
  assert llm_name in paths
  return dict(
        abbr=template.format(
          llm=llm_name, 
          pruner=pruner_type, 
          fuser=fuser_type, 
          memory=memory_type, 
          chunk=chunk_size, 
          compressed_chunk=compressed_chunk_size, 
          dq=d_query, 
          trained=is_trained),  # name in path
        type=PrunerModel, 
        pe_config=pe_config,
        compresser_config=compresser_config,
        model_name_or_path=paths[llm_name],
        collie_config_path=paths[llm_name],
        tokenizer_only=False,
        long_bench_cat=-1,
        extract_pred_after_decode=True,
        tokenizer_name_or_path=paths[llm_name], 
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left',
                              trust_remote_code=True, use_fast=False, ),
        max_out_len=128,
        max_seq_len=16384,
        batch_size=1, 
        run_cfg=dict(num_gpus=num_gpus[llm_name], num_procs=num_gpus[llm_name]),  # tp or pp size
    )

all_models = {}
for llm in paths.keys():
  for chunk_size in chunk_sizes:
    for fuser_type in fuser_types:
      for pruner_type in pruner_types:
        for memory_type in memory_types:
          for compressed_chunk_size in compressed_chunk_sizes:
            for d_q in d_querys:
              for is_trained in (True, False):
                model_cfg = build_model_cfg(
                    llm_name=llm, chunk_size=chunk_size, d_query=d_q, compressed_chunk_size=compressed_chunk_size, pruner_type=pruner_type, 
                    fuser_type=fuser_type, memory_type=memory_type,  is_trained=is_trained
                )
                all_models[model_cfg["abbr"]] = model_cfg