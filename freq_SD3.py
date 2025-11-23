import torch
import torch.nn.functional as F
from frequency_util import freq_com_SD3
  
    
def register_time(model, t):

    for block in range(0, 24):
        module = model.transformer_blocks[block].attn
        setattr(module, 't', t)
    for block in range(0, 13):
        module = model.transformer_blocks[block].attn2
        setattr(module, 't', t)

def register_FIJ(model, injection_schedule):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None):

            residual = hidden_states

            batch_size = hidden_states.shape[0]

            # `sample` projections.
            query = self.to_q(hidden_states)
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // self.heads

            query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            
            if encoder_hidden_states is not None and self.injection_schedule is not None and (
                self.t in self.injection_schedule or self.t == 1000.):
                B = batch_size // 4
               
                query[B*2:B*3] = query[:B]
                query[B*3:B*4] = query[:B]

                key[B*2:B*3] = key[:B]
                key[B*3:B*4] = key[:B]
                  
                value[B*2:B*3] = value[:B]
                value[B*3:B*4] = value[:B]

                encoder_hidden_states[B*2:B*3] = encoder_hidden_states[:B]
                encoder_hidden_states[B*3:B*4] = encoder_hidden_states[:B]

            if self.norm_q is not None:
                query = self.norm_q(query)
            if self.norm_k is not None:
                key = self.norm_k(key)

            # `context` projections.
            if encoder_hidden_states is not None:

                encoder_hidden_states_query_proj = self.add_q_proj(encoder_hidden_states)
                encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
                encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

                encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                    batch_size, -1, self.heads, head_dim
                ).transpose(1, 2)
                encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                    batch_size, -1, self.heads, head_dim
                ).transpose(1, 2)
                encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                    batch_size, -1, self.heads, head_dim
                ).transpose(1, 2)

                if self.norm_added_q is not None:
                    encoder_hidden_states_query_proj = self.norm_added_q(encoder_hidden_states_query_proj)
                if self.norm_added_k is not None:
                    encoder_hidden_states_key_proj = self.norm_added_k(encoder_hidden_states_key_proj)
                                       

                query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
                key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
                value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            if encoder_hidden_states is not None:
                # Split the attention outputs.
                hidden_states, encoder_hidden_states = (
                    hidden_states[:, : residual.shape[1]],
                    hidden_states[:, residual.shape[1] :],
                )
                if not self.context_pre_only:
                    encoder_hidden_states = self.to_add_out(encoder_hidden_states)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)

            if encoder_hidden_states is not None:
                return hidden_states, encoder_hidden_states
            else:
                return hidden_states
            
        return forward

    for block in range(13,24): # cross-attention
        module = model.transformer_blocks[block].attn
        module.forward = sa_forward(module)
        setattr(module, 'injection_schedule', injection_schedule)

def register_FRI(model, injection_schedule):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None):

            residual = hidden_states

            batch_size = hidden_states.shape[0]

            # `sample` projections.
            query = self.to_q(hidden_states)
            key = self.to_k(hidden_states)
            value = self.to_v(hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // self.heads

            query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

            if self.norm_q is not None:
                query = self.norm_q(query)
            if self.norm_k is not None:
                key = self.norm_k(key)
                
            B = batch_size // 3

            query[B*2:B*3] = freq_com_SD3(query[:B], query[B*2:B*3], alpha=1)
            query[B*3:B*4] = freq_com_SD3(query[:B], query[B*3:B*4], alpha=1)
            key[B*2:B*3] = freq_com_SD3(key[:B], key[B*2:B*3], alpha=1)
            key[B*3:B*4] = freq_com_SD3(key[:B], key[B*3:B*4], alpha=1)
            
            value[B*2:B*3] = value[:B]
            value[B*3:B*4] = value[:B]
            
            # `context` projections.
            if encoder_hidden_states is not None:
                encoder_hidden_states_query_proj = self.add_q_proj(encoder_hidden_states)
                encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
                encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

                encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                    batch_size, -1, self.heads, head_dim
                ).transpose(1, 2)
                encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                    batch_size, -1, self.heads, head_dim
                ).transpose(1, 2)
                encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                    batch_size, -1, self.heads, head_dim
                ).transpose(1, 2)

                if self.norm_added_q is not None:
                    encoder_hidden_states_query_proj = self.norm_added_q(encoder_hidden_states_query_proj)
                if self.norm_added_k is not None:
                    encoder_hidden_states_key_proj = self.norm_added_k(encoder_hidden_states_key_proj)
                    
                if self.injection_schedule is not None and (
                    self.t in self.injection_schedule or self.t == 1000.):
                    B = batch_size // 3
                    
                    query[B*2:B*3] = query[:B]
                    encoder_hidden_states_query_proj[B*2:B*3] = encoder_hidden_states_query_proj[:B]
                    query[B*3:B*4] = query[:B]
                    encoder_hidden_states_query_proj[B*3:B*4] = encoder_hidden_states_query_proj[:B]
                    
                    key[B*2:B*3] = key[:B]
                    encoder_hidden_states_key_proj[B*2:B*3] = encoder_hidden_states_key_proj[:B]
                    key[B*3:B*4] = key[:B]
                    encoder_hidden_states_key_proj[B*3:B*4] = encoder_hidden_states_key_proj[:B]
                                        
                    value[B*2:B*3] = value[:B]
                    encoder_hidden_states_value_proj[B*2:B*3] = encoder_hidden_states_value_proj[:B]
                    value[B*3:B*4] = value[:B]
                    encoder_hidden_states_value_proj[B*3:B*4] = encoder_hidden_states_value_proj[:B]
                    

                query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
                key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
                value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

            hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            if encoder_hidden_states is not None:
                # Split the attention outputs.
                hidden_states, encoder_hidden_states = (
                    hidden_states[:, : residual.shape[1]],
                    hidden_states[:, residual.shape[1] :],
                )
                if not self.context_pre_only:
                    encoder_hidden_states = self.to_add_out(encoder_hidden_states)

            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)

            if encoder_hidden_states is not None:
                return hidden_states, encoder_hidden_states
            else:
                return hidden_states
            
        return forward

    for block in range(13): # self-attention
        module = model.transformer_blocks[block].attn2
        module.forward = sa_forward(module)
        setattr(module, 'injection_schedule', injection_schedule)

def FIA(pipe, FIJ_injection_t=51, FRI_injection_t=51):
    
    FRI_injection_timesteps = pipe.scheduler.timesteps[:FRI_injection_t]
    FIJ_injection_timesteps = pipe.scheduler.timesteps[:FIJ_injection_t]
    register_FIJ(pipe.transformer, FIJ_injection_timesteps)
    register_FRI(pipe.transformer, FRI_injection_timesteps)
    
    return pipe