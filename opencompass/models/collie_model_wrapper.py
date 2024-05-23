import os
import sys

import argparse

from functools import reduce
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger

from opencompass.models.base import BaseModel, LMTemplateParser

from typing import Dict, List, Optional, Union
from torch.cuda.amp import autocast

# sys.path.append('/cpfs01/shared/public/lvkai/workspace/collie/')    # FIXME: remove this line
sys.path.append('/remote-home/mqhuang/kv-cache/collie/collie/')

from collie import CollieConfig
from transformers.generation.utils import GenerationConfig
from transformers import AutoTokenizer


def str_to_bool(value: str):
    if value.lower() in ['false', 'f', '0', 'no', 'n']:
        return False
    elif value.lower() in ['true', 't', '1', 'yes', 'y']:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def get_namespace_from_dict_for_sink(config: dict):
    
    config = list(reduce(lambda i, j :i + j, [(key, config[key]) for key in config]))

    config = ['--' + item if i % 2 == 0 else str(item) for i, item in enumerate(config)]
    
    parser = argparse.ArgumentParser(description='define running config')

    parser.add_argument('--exp', type=str_to_bool, default=False)
    parser.add_argument('--log', type=str_to_bool, default=False)

    parser.add_argument('--log_base', type=float, default=torch.e)
    parser.add_argument('--log_clip', type=float, default=1)
    parser.add_argument('--exp_base', type=float, default=512.)
    parser.add_argument('--max_length', type=int, default=512)

    parser.add_argument('--streaming_enable', type=str_to_bool, default=False)
    parser.add_argument('--streaming_stride', type=int, default=1)
    parser.add_argument('--start_size', type=int, default=4)
    parser.add_argument('--local_size', type=int, default=512)
    parser.add_argument('--memory_option', type=str, default='sink')
    parser.add_argument('--memory_length', type=int, default=1)

    parser.add_argument('--base', type=float, default=10000.)
    parser.add_argument('--pi_lambda', type=float, default=1.)

    parser.add_argument('--ntk_option', type=str, default='none', choices=['none', 'fixed', 'dynamic'])
    parser.add_argument('--ntk_alpha', type=float, default=1.)
    
    return parser.parse_args(config)

class PrunerModel(BaseModel):
    
    def __init__(
        self,
        collie_config_path: str,
        pe_config: Dict, 
        compresser_config: Dict,
        model_name_or_path: str,
        max_seq_len: int=4096,
        tokenizer_only: bool=False,
        long_bench_cat: int=-1,
        batch_padding: bool=True,
        extract_pred_after_decode: bool=True,
        tokenizer_name_or_path: str=None,
        tokenizer_kwargs: Dict={},
        prompt_format: str='{prompt}',
        meta_template: Optional[Dict]=None,
    ):
        super().__init__(path=collie_config_path,
                         max_seq_len=max_seq_len,
                         tokenizer_only=tokenizer_only,
                         meta_template=meta_template)
        config = CollieConfig.from_pretrained(collie_config_path, trust_remote_code=True, dtype=torch.float16)
        
        config.use_flash = True
        config.ds_config = {
            'bf16': {
                'enabled': True,
            },
        }
        
        config.tp_size = int(os.environ['WORLD_SIZE'])
        config.dp_size = 1
        config.model_config.num_hidden_layers = compresser_config.get('num_hidden_layers', config.model_config.num_hidden_layers)
        config.checkpointing = True
        
        chunk_size = compresser_config.get('chunk_size', 512)
        d_query = compresser_config.get('d_query', config.hidden_size // 4)
        compressed_chunk_size = compresser_config.get('compressed_chunk_size', chunk_size // 8)
        num_sink_tokens = compresser_config.get('num_sink_tokens', 4)
        pruner_type = compresser_config.get('pruner_type', None)
        fuser_type = compresser_config.get('fuser_type', None)
        memory_type = compresser_config.get('memory_type', None)
        perceiver_path = compresser_config.get('perceiver_path', None)
        assert pruner_type is not None or fuser_type is not None
        # set pe_config
        self.pe_config = pe_config
        config.pe_config = pe_config
        
        d_model=config.hidden_size // config.num_attention_heads * config.num_key_value_heads
        num_heads=config.num_key_value_heads
        num_layers=config.num_hidden_layers
        
        mem_perceiver_config = {
            # llm config
            "d_model": d_model,
            "num_heads": num_heads,
            "num_layers": num_layers,
            # custom config
            "query_len": compressed_chunk_size,
            "d_query": d_query,
            "compressed_chunk_size": compressed_chunk_size,
            "chunk_size": chunk_size,
            "num_sink_tokens": num_sink_tokens,
            "memory_type": memory_type,
        }
        # init collie model config
        setattr(config, 'mem_perceiver_config', mem_perceiver_config)
        
        if tokenizer_name_or_path:
            self._load_tokenizer(tokenizer_name_or_path=tokenizer_name_or_path, tokenizer_kwargs=tokenizer_kwargs)
        else:
            self._load_tokenizer(tokenizer_name_or_path=model_name_or_path, tokenizer_kwargs=tokenizer_kwargs)
        
        if not tokenizer_only:
            self._load_model(
                model_name_or_path=model_name_or_path,
                pruner_type=pruner_type,
                fuser_type=fuser_type,
                config=config,
                perceiver_path=perceiver_path
            )
            
        self.long_bench_cat = long_bench_cat
        self.batch_padding = batch_padding
        self.extract_pred_after_decode = extract_pred_after_decode
        self.prompt_format = prompt_format
        self.meta_template = meta_template
        self.max_seq_len = max_seq_len
        
        self.logger = get_logger()
        self.model = self.model.to(torch.bfloat16)
        
        # set eval
        self.model.eval().cuda()
    
    def _load_model(
        self,     
        model_name_or_path: str,
        pruner_type: str,
        fuser_type: str,
        config: CollieConfig,
        perceiver_path: str=None,
    ):
        from collie.models.mem_perceiver import AutoPruner, AutoFuser
        if pruner_type is not None:
            self.model = AutoPruner.from_pretrained(
                pruner_type=pruner_type,
                config=config,
                pretrained_model_name_or_path=model_name_or_path,
                perceiver_path=perceiver_path
            )
        elif fuser_type is not None:
            self.model = AutoFuser.from_pretrained(
                fuser_type=fuser_type,
                config=config,
                pretrained_model_name_or_path=model_name_or_path,
                perceiver_path=perceiver_path
            )
    
    def _load_tokenizer(self, tokenizer_name_or_path: Optional[str], tokenizer_kwargs: dict):
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)  # , local_files_only=True
        
    def generate(self, inputs: List[str], max_out_len: int, **kwargs) -> List[str]:
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else eos_token_id
        num_beams = kwargs.get('num_beams', 1)
        do_sample = kwargs.get('do_sample', False)
        use_cache = kwargs.get('use_cache', True)
        generation_config = GenerationConfig(
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            num_beams=num_beams, do_sample=do_sample, use_cache=use_cache
        )
        if self.batch_padding and len(inputs) > 1:
            return self._batch_generate(inputs=inputs, max_out_len=max_out_len, generation_config=generation_config)
        else:
            return sum(
                (
                    self._single_generate(
                        inputs=[input_], max_out_len=max_out_len, generation_config=generation_config
                    ) for input_ in inputs
                ),
                []
            )
        
    def _batch_generate(self, inputs: List[str], max_out_len: int, generation_config: GenerationConfig) -> List[str]:
        if self.extract_pred_after_decode:
            prompt_lens = [len(input_) for input_ in inputs]

        # step-1: tokenize the input with batch_encode_plus
        tokens = self.tokenizer.batch_encode_plus(inputs, padding=True, truncation=True, 
                                                  max_length=self.max_seq_len - max_out_len)
        tokens = {
            k: torch.tensor(np.array(tokens[k]), device=self.model.device)
            for k in tokens if k in ['input_ids', 'attention_mask']
        }

        outputs = self.model.generate(**tokens, generation_config=generation_config)

        if not self.extract_pred_after_decode:
            outputs = outputs[:, tokens['input_ids'].shape[1]:]
        # self.logger.info('outputs return')
        decodeds = self.tokenizer.batch_decode(outputs.cpu().tolist(), skip_special_tokens=True)
        
        if self.extract_pred_after_decode:
            decodeds = [
                token[len_:] for token, len_ in zip(decodeds, prompt_lens)
            ]
        
        return decodeds
    
    def _single_generate(self, inputs: List[str], max_out_len: int, generation_config: GenerationConfig) -> List[str]:
        if self.extract_pred_after_decode:
            prompt_lens = [len(input_) for input_ in inputs]
            
        if self.long_bench_cat > 0:
            inputs = [self.prompt_format.format(prompt=prompt) for prompt in inputs]
            input_ids = self.tokenizer(inputs, padding=False, truncation=False)['input_ids']
            input_ids = torch.tensor(input_ids)
            if input_ids.shape[-1] > self.long_bench_cat:
                input_ids = torch.cat([input_ids[:, : self.long_bench_cat // 2], input_ids[:, - self.long_bench_cat // 2:]], dim=-1).to(device=self.model.device)
            else:
                input_ids = input_ids.to(device=self.model.device)
        elif self.pe_config.get('streaming_enable', False) and self.pe_config.get('memory_option', '') in ['', 'sink']:
            input_ids = self.tokenizer(inputs, padding=False, truncation=False)['input_ids']
            input_ids = torch.tensor(input_ids)
            if input_ids.shape[-1] > self.pe_config['start_size'] + self.pe_config['local_size']:
                input_ids = torch.cat([input_ids[:, : self.pe_config['start_size']], input_ids[:, - self.pe_config['local_size']:]], dim=-1).to(device=self.model.device)
            else:
                input_ids = input_ids.to(device=self.model.device)
        else:
            input_ids = self.tokenizer(inputs, padding=False, truncation=True, max_length=self.max_seq_len)['input_ids']
            input_ids = torch.tensor(input_ids).to(device=self.model.device)
        
        generation_config.max_new_tokens = max_out_len
        with autocast():
            outputs = self.model.generate(input_ids=input_ids, generation_config=generation_config)
        decodeds = self.tokenizer.batch_decode(outputs.cpu().tolist(), skip_special_tokens=True)
        if self.extract_pred_after_decode:
            decodeds = [
                token[len_:] for token, len_ in zip(decodeds, prompt_lens)
            ]
        return decodeds
    
    def get_logits(self, inputs: List[str]):
        if self.batch_padding and len(inputs) > 1:
            tokens = self.tokenizer(inputs, padding=True, truncation=True,
                                    max_length=self.max_seq_len)
            tokens = {
                k: torch.tensor(np.array(tokens[k]), device=self.model.device)
                for k in tokens if k in ['input_ids', 'attention_mask']
            }
            outputs = self.model(**tokens)
        else:
            if self.long_bench_cat > 0:
                inputs = [self.prompt_format.format(prompt=prompt) for prompt in inputs]
                input_ids = self.tokenizer(inputs, padding=False, truncation=False)['input_ids']
                input_ids = torch.tensor(input_ids)
                if input_ids.shape[-1] > self.long_bench_cat:
                    input_ids = torch.cat([input_ids[:, : self.long_bench_cat // 2], input_ids[:, - self.long_bench_cat // 2:]], dim=-1).to(device=self.model.device)
                elif self.pe_config.get('streaming_enable', False) and self.pe_config.get('memory_option', '') in ['', 'sink']:
                    input_ids = self.tokenizer(inputs, padding=False, truncation=False)['input_ids']
                    input_ids = torch.tensor(input_ids)
                    if input_ids.shape[-1] > self.pe_config['start_size'] + self.pe_config['local_size']:
                        input_ids = torch.cat([input_ids[:, : self.pe_config['start_size']], input_ids[:, - self.pe_config['local_size']:]], dim=-1).to(device=self.model.device)
                    else:
                        input_ids = input_ids.to(device=self.model.device)
                else:
                    input_ids = input_ids.to(device=self.model.device)
            else:
                input_ids = self.tokenizer(inputs, padding=False, truncation=True, max_length=self.max_seq_len)['input_ids']
                input_ids = torch.tensor(input_ids).to(device=self.model.device)
            tokens = {'input_ids': input_ids}
            outputs = self.model(input_ids)
        return outputs.get('logits'), {'tokens': tokens}
    
    def get_ppl(self, inputs: List[str], mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """
        if self.batch_padding and len(inputs) > 1:
            assert self.tokenizer.pad_token
            return self._get_ppl(inputs, mask_length=mask_length)
        else:
            return np.concatenate([
                self._get_ppl(inputs=[text], mask_length=mask_length)
                for text in inputs
            ])
    
    def _get_ppl(self, inputs: List[str], mask_length: Optional[List[int]] = None) -> List[float]:
        outputs, inputs = self.get_logits(inputs)
        shift_logits = outputs[..., :-1, :].contiguous()
        
        shift_labels = inputs['tokens']['input_ids'][..., 1:].contiguous()
        
        # FIXME: if need to set the pad_token_id?
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)).view(shift_labels.size())
        
        if mask_length is not None:
            mask = torch.zeros_like(shift_labels)   # [ batch, seqlen ]
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask
        
        lens = (inputs['tokens']['input_ids'] != self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        if mask_length is not None:
            lens -= np.array(mask_length)
        loss = loss.float()
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
        return ce_loss
    
    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        return super().get_token_len(prompt)
        
    
# @MODELS.register_module()
class CollieModel(BaseModel):
    """Model wrapper around HuggingFace CausalLM.

    Args:
        path (str): The name or path to HuggingFace's model.
        hf_cache_dir: Set the cache dir to HF model cache dir. If None, it will
            use the env variable HF_MODEL_HUB. Defaults to None.
        max_seq_len (int): The maximum length of the input sequence. Defaults
            to 2048.
        tokenizer_path (str): The path to the tokenizer. Defaults to None.
        tokenizer_kwargs (dict): Keyword arguments for the tokenizer.
            Defaults to {}.
        peft_path (str, optional): The name or path to the HuggingFace's PEFT
            model. If None, the original model will not be converted to PEFT.
            Defaults to None.
        tokenizer_only (bool): If True, only the tokenizer will be initialized.
            Defaults to False.
        model_kwargs (dict): Keyword arguments for the model, used in loader.
            Defaults to dict(device_map='auto').
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        batch_padding (bool): If False, inference with be performed in for-loop
            without batch padding.
    """

    def __init__(self,
                 pe_config: dict,
                 config_path: str,
                 model_path: str,
                 model_type: str = 'pe', 
                 max_seq_len: int = 4096, 
                 long_bench_cat: int = -1, 
                 prompt_format: str = '{prompt}', 
                 tokenizer_kwargs: dict = dict(),
                 meta_template: Optional[Dict] = None,
                 extract_pred_after_decode: bool = False,
                 batch_padding: bool = False):
        
        super().__init__(path=config_path,
                         max_seq_len=max_seq_len,
                         tokenizer_only=False,
                         meta_template=meta_template)
        
        config = CollieConfig.from_pretrained(config_path, trust_remote_code=True)  # , local_files_only=True

        config.model_config.use_cache = True
        config.checkpointing = True
        config.use_flash = True
        config.ds_config = {
            'bf16': {
                'enabled': True,
            },
        }
        config.tp_size = int(os.environ['WORLD_SIZE'])
        config.dp_size = 1

        self.config = config
        self.model_path = model_path
        self.batch_padding = batch_padding
        self.long_bench_cat = long_bench_cat
        self.prompt_format = prompt_format
        
        self._load_tokenizer(tokenizer_path=config_path, tokenizer_kwargs=tokenizer_kwargs)
        self._load_model(model_path=model_path, config=config, pe_config=pe_config, model_type=model_type)
        self.pe_config = pe_config
        self.logger = get_logger()
        
        self.extract_pred_after_decode = extract_pred_after_decode
        
        self.generation_config = GenerationConfig(
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            num_beams=1, do_sample=False, use_cache=True
        )

    def _load_tokenizer(self, tokenizer_path: Optional[str], tokenizer_kwargs: dict):
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, **tokenizer_kwargs)  # , local_files_only=True
        self.eos_token_id = self.tokenizer.eos_token_id
        self.tokenizer.bos_token = '<s>'
        self.tokenizer.eos_token = '</s>'
        self.tokenizer.pad_token_id = 0
        self.pad_token_id = 0
            
    def _load_model(self, model_path: str, config: CollieConfig, pe_config: dict, model_type: str):
        
        if model_type == 'pe':
            from opencompass.models.collie_model_with_pe import LlamaForCausalLM
            config.__setattr__('pe_config', pe_config)
        elif model_type == 'sink':
            from opencompass.models.collie_model_with_sink import LlamaForCausalLM       
            args = get_namespace_from_dict_for_sink(pe_config)
            config.__setattr__('args', args)
        
        if model_type == 'other':
            from collie import setup_distribution
            from transformers import AutoModel, AutoModelForCausalLM, PretrainedConfig

            setup_distribution(config=config)
            model_config = config.model_config
            model_config._flash_attn_2_enabled = True
            model_config.attn_implementation = "flash_attention_2"
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_path, config=model_config, torch_dtype=torch.float16, 
                                                                  trust_remote_code=True)  # , local_files_only=True
            except ValueError:
                self.model = AutoModel.from_pretrained(model_path, config=model_config, torch_dtype=torch.float16, 
                                                       trust_remote_code=True)
                                                      
        
        elif model_path.__contains__(':'):
            self.model = LlamaForCausalLM.from_pretrained(model_path_or_name=model_path, 
                                                          protocol='petrel', config=config, trust_remote_code=True)
        else:
            self.model = LlamaForCausalLM.from_pretrained(model_path_or_name=model_path, 
                                                          config=config, trust_remote_code=True)

        # use bf16
        self.model = self.model.half()

        self.model.eval().cuda()
        
        # if model_type == 'sink' and 'attn' in pe_config['memory_option']:
        #     import deepspeed
        #     with deepspeed.zero.GatheredParameters([layer.self_attn["q_proj"].weight for layer in self.model.model.layers] + 
        #                                            [layer.self_attn["k_proj"].weight for layer in self.model.model.layers] + 
        #                                            [layer.self_attn["v_proj"].weight for layer in self.model.model.layers]):   
        #         for layer in self.model.model.layers:
        #             layer.build()

    def generate(self, inputs: List[str], max_out_len: int, **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        if self.batch_padding and len(inputs) > 1:
            return self._batch_generate(inputs=inputs, max_out_len=max_out_len, **kwargs)
        else:
            return sum((self._single_generate(
                inputs=[input_], max_out_len=max_out_len, **kwargs)
                for input_ in inputs), [])

    def _batch_generate(self, inputs: List[str], max_out_len: int, **kwargs) -> List[str]:
        """Support for batch prompts inference.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        if self.extract_pred_after_decode:
            prompt_lens = [len(input_) for input_ in inputs]

        # step-1: tokenize the input with batch_encode_plus
        tokens = self.tokenizer.batch_encode_plus(inputs, padding=True, truncation=True, 
                                                  max_length=self.max_seq_len - max_out_len)
        tokens = {
            k: torch.tensor(np.array(tokens[k]), device=self.model.device)
            for k in tokens if k in ['input_ids', 'attention_mask']
        }

        # step-2: conduct model forward to generate output
        generation_config = self.generation_config
        generation_config.max_new_tokens = max_out_len
        # self.logger.info('input_ids given')
        outputs = self.model.generate(**tokens, generation_config=generation_config)

        if not self.extract_pred_after_decode:
            outputs = outputs[:, tokens['input_ids'].shape[1]:]
        # self.logger.info('outputs return')
        decodeds = self.tokenizer.batch_decode(outputs.cpu().tolist(), skip_special_tokens=True)

        if self.extract_pred_after_decode:
            decodeds = [
                token[len_:] for token, len_ in zip(decodeds, prompt_lens)
            ]

        return decodeds

    def _single_generate(self, inputs: List[str], max_out_len: int, **kwargs) -> List[str]:
        """Support for single prompt inference.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        if self.extract_pred_after_decode:
            prompt_lens = [len(input_) for input_ in inputs]

        if self.long_bench_cat > 0:
            inputs = [self.prompt_format.format(prompt=prompt) for prompt in inputs]
            input_ids = self.tokenizer(inputs, padding=False, truncation=False)['input_ids']
            input_ids = torch.tensor(input_ids)
            if input_ids.shape[-1] > self.long_bench_cat:
                input_ids = torch.cat([input_ids[:, : self.long_bench_cat // 2], input_ids[:, - self.long_bench_cat // 2:]], dim=-1).to(device=self.model.device)
            else:
                input_ids = input_ids.to(device=self.model.device)
        elif self.pe_config.get('streaming_enable', False) and self.pe_config.get('memory_option', '') in ['', 'sink']:
            input_ids = self.tokenizer(inputs, padding=False, truncation=False)['input_ids']
            input_ids = torch.tensor(input_ids)
            if input_ids.shape[-1] > self.pe_config['start_size'] + self.pe_config['local_size']:
                input_ids = torch.cat([input_ids[:, : self.pe_config['start_size']], input_ids[:, - self.pe_config['local_size']:]], dim=-1).to(device=self.model.device)
            else:
                input_ids = input_ids.to(device=self.model.device)
        else:
            input_ids = self.tokenizer(inputs, padding=False, truncation=True, max_length=self.max_seq_len)['input_ids']
            input_ids = torch.tensor(input_ids).to(device=self.model.device)
        
        generation_config = self.generation_config
        generation_config.max_new_tokens = max_out_len
        # self.logger.info('input_ids give')
        outputs = self.model.generate(input_ids=input_ids, generation_config=generation_config)

        if not self.extract_pred_after_decode:
            outputs = outputs[:, input_ids.shape[1]:]
        # self.logger.info('outputs return')
        decodeds = self.tokenizer.batch_decode(outputs.cpu().tolist(), skip_special_tokens=True)

        if self.extract_pred_after_decode:
            decodeds = [
                token[len_:] for token, len_ in zip(decodeds, prompt_lens)
            ]

        return decodeds

    def get_logits(self, inputs: List[str]):

        if self.batch_padding and len(inputs) > 1:
            # batch inference
            tokens = self.tokenizer(inputs, padding=True, truncation=True,
                                    max_length=self.max_seq_len)

            tokens = {
                k: torch.tensor(np.array(tokens[k]), device=self.model.device)
                for k in tokens if k in ['input_ids', 'attention_mask']
            }
            outputs = self.model(**tokens)

        else:
            if self.long_bench_cat > 0:
                inputs = [self.prompt_format.format(prompt=prompt) for prompt in inputs]
                input_ids = self.tokenizer(inputs, padding=False, truncation=False)['input_ids']
                input_ids = torch.tensor(input_ids)
                if input_ids.shape[-1] > self.long_bench_cat:
                    input_ids = torch.cat([input_ids[:, : self.long_bench_cat // 2], input_ids[:, - self.long_bench_cat // 2:]], dim=-1).to(device=self.model.device)
                elif self.pe_config.get('streaming_enable', False) and self.pe_config.get('memory_option', '') in ['', 'sink']:
                    input_ids = self.tokenizer(inputs, padding=False, truncation=False)['input_ids']
                    input_ids = torch.tensor(input_ids)
                    if input_ids.shape[-1] > self.pe_config['start_size'] + self.pe_config['local_size']:
                        input_ids = torch.cat([input_ids[:, : self.pe_config['start_size']], input_ids[:, - self.pe_config['local_size']:]], dim=-1).to(device=self.model.device)
                    else:
                        input_ids = input_ids.to(device=self.model.device)
                else:
                    input_ids = input_ids.to(device=self.model.device)
            else:
                input_ids = self.tokenizer(inputs, padding=False, truncation=True, max_length=self.max_seq_len)['input_ids']
                input_ids = torch.tensor(input_ids).to(device=self.model.device)
            tokens = {'input_ids': input_ids}

            outputs = self.model(input_ids)
        return outputs.get('logits'), {'tokens': tokens}

    def get_ppl(self,
                inputs: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """

        if self.batch_padding and len(inputs) > 1:
            assert self.tokenizer.pad_token
            return self._get_ppl(inputs, mask_length=mask_length)
        else:
            return np.concatenate([
                self._get_ppl(inputs=[text], mask_length=mask_length)
                for text in inputs
            ])

    def _get_ppl(self,
                 inputs: List[str],
                 mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """

        outputs, inputs = self.get_logits(inputs)
        shift_logits = outputs[..., :-1, :].contiguous()

        shift_labels = inputs['tokens']['input_ids'][..., 1:].contiguous()

        self.tokenizer.pad_token_id = 0
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=0)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)).view(shift_labels.size())

        if mask_length is not None:
            mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask

        lens = (inputs['tokens']['input_ids'] !=
                0).sum(-1).cpu().numpy()
        if mask_length is not None:
            lens -= np.array(mask_length)
        loss = loss.float()
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
        return ce_loss

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        return len(self.tokenizer.encode(prompt))

