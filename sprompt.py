# @Time    : 2023/12/15 15:33
# @Author  : yxL
# @File    : sprompt.py
# @Software: PyCharm
# @Description :

import torch
import torch.nn as nn


class S_EPrompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, prompt_init='uniform', prompt_pool=False,
                 pool_size=None, num_layers=1, use_prefix_tune_for_e_prompt=False, num_heads=-1, same_key_value=False,
                 **kwargs):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool  # 是否使用提示池
        self.prompt_init = prompt_init
        self.pool_size = pool_size  # 提示池大小 即 已学任务数量(测试时)

        self.num_layers = num_layers  # 插入prompt的层数
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value

        # 创建提示池 (num_layers, pv_pk=2, pool_size, length, num_heads, embed_dim // num_heads)
        if self.prompt_pool:
            # user prefix style
            if self.use_prefix_tune_for_e_prompt:
                assert embed_dim % self.num_heads == 0
                if self.same_key_value:  # pk=pv       ↓ 只要1维度
                    prompt_pool_shape = (
                    self.num_layers, 1, self.pool_size, self.length, self.num_heads, embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':
                        self.prompt = nn.Parameter(torch.randn(
                            prompt_pool_shape))  # num_layers, 1, pool_size, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.prompt, -1, 1)
                    self.prompt = self.prompt.repeat(1, 2, 1, 1, 1,1)  # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                else:
                    prompt_pool_shape = (
                    self.num_layers, 2, self.pool_size, self.length, self.num_heads, embed_dim // self.num_heads)
                    if prompt_init == 'zero':
                        self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                    elif prompt_init == 'uniform':  # ↓ 2为 pk,pv 维度  p=[pk;pv]
                        self.prompt = nn.Parameter(torch.randn(prompt_pool_shape)) # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
                        nn.init.uniform_(self.prompt, -1, 1)
            else:  # 使用 prompt tuning
                # num_layers, pool_size, length*2, embed_dim 长度x2保持与prefix_tune一致
                prompt_pool_shape = (self.num_layers, self.pool_size, self.length*2, embed_dim)
                if prompt_init == 'zero':
                    self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                elif prompt_init == 'uniform':
                    self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                    nn.init.uniform_(self.prompt, -1, 1)

    def forward(self, idx=None, x_embed=None):
        out = dict()

        # if train:
        #     # 扩展task_id到batch_size
        #     idx = task_id.unsqueeze(0).expand(x_embed.shape[0])
        # else:
        #     # 计算cls_features与all_keys的L1距离
        #     idx = self.get_idx(all_keys,cls_features)

        if self.use_prefix_tune_for_e_prompt:
            batched_prompt_raw = self.prompt[:, :, idx]  # num_layers, 2, B, length, num_heads, embed_dim // num_heads

            num_layers, dual, batch_size, length, num_heads, heads_embed_dim = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(
                num_layers, batch_size, dual, length, num_heads, heads_embed_dim
            )
        else:
            batched_prompt_raw = self.prompt[:, idx]
            num_layers, batch_size, length, embed_dim = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(
                num_layers, batch_size, length, embed_dim
            )

        out['batched_prompt'] = batched_prompt # num_layers, B, pv_pk=2, length, num_heads, embed_dim // num_heads

        return out

if __name__ == '__main__':
    prompt = S_EPrompt(prompt_pool=True,pool_size=10,num_layers=3,use_prefix_tune_for_e_prompt=True,num_heads=12)
    batch_size = 64
    image_token = 196
    output = prompt(idx=torch.tensor(2).unsqueeze(0).expand(batch_size),
                    x_embed=torch.randn(batch_size,image_token,768),)
    print(output['batched_prompt'].shape)