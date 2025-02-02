
from opencompass.models.collie_model_wrapper import CollieModel

import numpy as np

paths = {'llama2_7B': '/remote-home/share/models/llama_v2_hf/7b', 
         'llama2_13B': '/remote-home/share/models/llama_v2_hf/13b', 
         }


num_gpus = {'llama2_7B': 1, 'llama2_13B': 2, }

# root = 'p_ssd:s3://P_model_weights/liuxiaoran/FEPE-collie/checkpoints'
root = '/remote-home/share/models/llama_v2_hf/7b'

tags = [       
        ('-base_10000', 'llama2_7B', 'rope_inv_2d_raw', 
         {'exp': False, '1d': False, 'imp': False, 'log': False, 
          'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
          'pi_lambda': 1, 'base': 10000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('', 'llama2_7B', 'llama2_7B', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 10000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_100', 'llama2_7B', 'hang_100', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 100.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_500', 'llama2_7B', 'hang_500', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 500.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_500_log', 'llama2_7B', 'hang_500', 
#          {'exp': False, '1d': False, 'imp': False, 'log': True, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 500.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_652', 'llama2_7B', 'hang_652', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 652.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_652_log', 'llama2_7B', 'hang_652', 
#          {'exp': False, '1d': False, 'imp': False, 'log': True, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 652.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         # ('-base_1304', 'llama2_7B', 'hang_1304', 
#         #  {'exp': False, '1d': False, 'imp': False, 'log': False, 
#         #   'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#         #   'pi_lambda': 1, 'base': 1304.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         # ('-base_2608', 'llama2_7B', 'hang_2608', 
#         #  {'exp': False, '1d': False, 'imp': False, 'log': False, 
#         #   'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#         #   'pi_lambda': 1, 'base': 2608.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-ntk_fixed_8', 'llama2_7B', 'llama2_7B', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 10000.0, 'ntk_option': 'fixed', 'ntk_alpha': 8., }), 
#         ('-ntk_dynamic', 'llama2_7B', 'llama2_7B', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 10000.0, 'ntk_option': 'dynamic', 'ntk_alpha': 1., }), 
#         ('-base_40000', 'llama2_7B', 'hang_40000', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 40000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_80000', 'llama2_7B', 'hang_80000', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 80000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_120000', 'llama2_7B', 'hang_120000', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 120000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_160000', 'llama2_7B', 'hang_160000', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 160000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_400000', 'llama2_7B', 'hang_400000', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 400000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_600000', 'llama2_7B', 'hang_600000', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 600000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_1000000', 'llama2_7B', 'hang_1000000', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 1000000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_2000000', 'llama2_7B', 'hang_2000000', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 2000000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
        
#         ('-base_10000_16k', 'llama2_7B', 'hang_10000_16K', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 10000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_500_16k', 'llama2_7B', 'hang_500_16k', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 500.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_500_16k_log', 'llama2_7B', 'hang_500_16k', 
#          {'exp': False, '1d': False, 'imp': False, 'log': True, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 500.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_40000_16k', 'llama2_7B', 'hang_40000_16K', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 40000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_80000_16k', 'llama2_7B', 'hang_80000_16K', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 80000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_120000_16k', 'llama2_7B', 'hang_120000_16K', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 120000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_1000000_16k', 'llama2_7B', 'hang_1000000_16k', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 1000000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
    
#         ('-base_10000', 'llama2_13B', 'rope_inv_2d_raw', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 10000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('', 'llama2_13B', 'llama2_13B', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 10000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_100', 'llama2_13B', 'hang_100', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 100.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_500', 'llama2_13B', 'hang_500', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 500.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_500_log', 'llama2_13B', 'hang_500', 
#          {'exp': False, '1d': False, 'imp': False, 'log': True, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 500.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_652', 'llama2_13B', 'hang_652', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 652.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_652_log', 'llama2_13B', 'hang_652', 
#          {'exp': False, '1d': False, 'imp': False, 'log': True, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 652.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         # ('-base_1304', 'llama2_13B', 'hang_1304', 
#         #  {'exp': False, '1d': False, 'imp': False, 'log': False, 
#         #   'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#         #   'pi_lambda': 1, 'base': 1304.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         # ('-base_2608', 'llama2_13B', 'hang_2608', 
#         #  {'exp': False, '1d': False, 'imp': False, 'log': False, 
#         #   'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#         #   'pi_lambda': 1, 'base': 2608.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-ntk_fixed_8', 'llama2_13B', 'llama2_13B', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 10000.0, 'ntk_option': 'fixed', 'ntk_alpha': 8., }), 
#         ('-ntk_dynamic', 'llama2_13B', 'llama2_13B', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 10000.0, 'ntk_option': 'dynamic', 'ntk_alpha': 1., }), 
#         ('-base_40000', 'llama2_13B', 'hang_40000', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 40000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_80000', 'llama2_13B', 'hang_80000', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 80000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_120000', 'llama2_13B', 'hang_120000', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 120000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_160000', 'llama2_13B', 'hang_160000', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 160000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_400000', 'llama2_13B', 'hang_400000', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 400000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_600000', 'llama2_13B', 'hang_600000', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 600000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_1000000', 'llama2_13B', 'hang_1000000', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 1000000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         # ('-base_2000000', 'llama2_13B', 'hang_2000000', 
#         #  {'exp': False, '1d': False, 'imp': False, 'log': False, 
#         #   'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#         #   'pi_lambda': 1, 'base': 2000000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
        
#         ('-base_10000_16k', 'llama2_13B', 'hang_10000_16K', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 10000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_500_16k', 'llama2_13B', 'hang_500_16K', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 500.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_500_16k_log', 'llama2_13B', 'hang_500_16K', 
#          {'exp': False, '1d': False, 'imp': False, 'log': True, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 500.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_40000_16k', 'llama2_13B', 'hang_40000_16K', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 40000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_80000_16k', 'llama2_13B', 'hang_80000_16K', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 80000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_120000_16k', 'llama2_13B', 'hang_120000_16K', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 120000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
#         ('-base_1000000_16k', 'llama2_13B', 'hang_1000000_16k', 
#          {'exp': False, '1d': False, 'imp': False, 'log': False, 
#           'exp_base': 4096, 'log_base': 4096, 'log_clip': 1, 'max_length': 4096, 
#           'pi_lambda': 1, 'base': 1000000.0, 'ntk_option': 'none', 'ntk_alpha': 1., }), 
    ]


models = [
    dict(
        abbr='{}{}'.format(group, abbr),  # name in path
        type=CollieModel, 
        pe_config=pe_config,
        model_type='pe', 
        model_path=paths[group] if group in paths else '/remote-home/share/models/llama_v2_hf/7b',  # pytorch_model.bin
        # model_path=paths[path] if path in paths else 'p_ssd:s3://P_model_weights/liuxiaoran/FEPE-collie/checkpoints/pjlab_fepe_{}_4096-{}/epoch_1/'.format(group, path),  # pytorch_model.bin
        config_path=paths[group],
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left',
                              trust_remote_code=True, use_fast=False, ),
        max_out_len=128,
        max_seq_len=16384,
        batch_size=1, 
        batch_padding=True,
        run_cfg=dict(num_gpus=num_gpus[group], num_procs=num_gpus[group]),  # tp or pp size
    ) for abbr, group, path, pe_config in tags]

