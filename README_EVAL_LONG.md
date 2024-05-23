# Eval Compresser
eval config文件：`configs/eval_efficient_kv/eval_long_bench.py`

例如，评测基于llama2的perceiver:
```python
llm=["llama2_7B"]
fuser = [None]
pruner=['perceiver']
memory=["Incremental_Chunk_Streaming_Dynamic_History"]
chunk_size=[512]
compressed_chunk_size=[c_s // 8 for c_s in chunk_size]
dq=[1024]
is_trained=[False]
```

可选的参数在`configs/eval_efficient_kv/compresser.py`。例如可选的pruners:
```python
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
```

注意，需要修改`configs/eval_efficient_kv/compresser.py`中的`paths`变量，加上自己的llm路径

适配opencompass的模型文件在`opencompass/models/collie_model_wrapper.py`，必要时可以基于这个调整模型的评测。