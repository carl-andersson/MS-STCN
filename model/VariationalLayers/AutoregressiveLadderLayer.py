import torch

from model.base import LatentState
from model.torch_wrappers import ModuleWrapper
from model.VariationalLayers.LadderLayer import LayerOptions
from model.VariationalLayers.BaseLayer import BaseBuilder
from model.DistModules import *
from model.torch_ext_causal_conv import *


class AutoregressiveLayerOptions(LayerOptions):
    def __init__(self, latent_size, bypass_frac, ar_conn_frac, stride, inf_pre_options, inf_post_options,
                 gen_pre_options, gen_post_options, ar_conn_options, size=None, size_multiplier=8):
        super(AutoregressiveLayerOptions, self).__init__(latent_size, bypass_frac, inf_pre_options, inf_post_options,
                                                         gen_pre_options, gen_post_options, size, size_multiplier)

        self.ar_conn_frac = ar_conn_frac
        self.stride = stride

        self.ar_conn_options = ar_conn_options

    @property
    def ar_conn_size(self):
        return math.floor(self.size*self.ar_conn_frac)


class AutoregressiveLayerBuilder(AutoregressiveLayerOptions, BaseBuilder):

    def __init__(self, layer_options, prev_layer=None, input_size=None, output_size=None):
        super(AutoregressiveLayerBuilder, self).__init__(**layer_options)
        self.next_layer = None
        if prev_layer is not None:
            self.input_size = prev_layer.inf_output
            self.output_size = prev_layer.inf_output
            prev_layer.next_layer = self
        else:
            assert(input_size is not None)

            self.input_size = input_size
            if output_size is None:
                self.output_size = self.size
            else:
                self.output_size = output_size

    def build(self):
        if self.next_layer is not None:
            sublayer = self.next_layer.build()
        else:
            sublayer = None

        # Inf: Encode from below, more concrete, layer
        inf_pre = ModuleWrapper(input_size=self.input_size, output_size=self.size, downscale=self.stride,
                                **self.inf_pre_options)

        # Autoregressive connection
        ar_conn = ModuleWrapper(input_size=inf_pre.output_size, output_size=self.ar_conn_size,
                                **self.ar_conn_options)

        # Gen: decode from above, more abstract, layer if any
        gen_pre_size_in = self.size if sublayer is not None else 0
        gen_pre = ModuleWrapper(input_size=gen_pre_size_in, output_size=self.size,
                                in_channels_cond=ar_conn.output_size,
                                **self.gen_pre_options)

        if self.latent_size > 0:
            # Latent q
            latent_q_size_in = inf_pre.output_size + gen_pre.output_size
            latent_q = DistributionLayer(latent_q_size_in, self.latent_size,
                                         NormalDist, {},
                                         CausalConvMLP1d, {'n_layers': 1})

            # Latent p, if None then unit Normal dist will be used instead

            if gen_pre.output_size > 0:
                latent_p = DistributionLayer(gen_pre.output_size, self.latent_size,
                                             NormalDist, {},
                                             CausalConvMLP1d, {'n_layers': 1})
            else:
                latent_p = None
        else:
            latent_p = None
            latent_q = None

        # Decode the latent sample
        latent_dec = ModuleWrapper(SkipConnection1D, self.latent_size, self.latent_size)

        # Skip the latent layer
        bypass = ModuleWrapper(SkipConnection1D, gen_pre.output_size, self.bypass_size)

        # Inf: Further encode the data, if needed
        inf_post_size_out = self.inf_output if self.next_layer is not None else 0
        inf_post = ModuleWrapper(input_size=inf_pre.output_size, output_size=inf_post_size_out,
                                 **self.inf_post_options)

        # Gen: Decode to next Ladder layer
        gen_post = ModuleWrapper(input_size=bypass.output_size, output_size=self.output_size, upscale=self.stride,
                                 in_channels_cond=latent_dec.output_size,
                                 **self.gen_post_options)

        ladder_layer = AutoregressiveLadderLayer(gen_pre=gen_pre.get_module(), gen_post=gen_post.get_module(),
                                                 bypass=bypass.get_module(), ar_conn=ar_conn.get_module(),
                                                 inf_pre=inf_pre.get_module(), inf_post=inf_post.get_module(),
                                                 latent_q=latent_q, latent_p=latent_p, latent_dec=latent_dec.get_module(),
                                                 stride=self.stride,
                                                 sublayer=sublayer)

        return ladder_layer


class AutoregressiveLadderLayer(nn.Module):

    def __init__(self, gen_pre, gen_post, bypass, ar_conn,
                 inf_pre, inf_post, latent_q, latent_p, latent_dec,
                 stride, sublayer=None):
        super().__init__()
        self.gen_pre = gen_pre
        self.gen_post = gen_post
        self.bypass = bypass
        self.ar_conn = ar_conn

        self.inf_pre = inf_pre
        self.inf_post = inf_post
        self.latent_q = latent_q
        self.latent_p = latent_p
        self.latent_dec = latent_dec

        self.sublayer = sublayer

        self.gen_batchnorm = None
        self.inf_batchnorm = None

        self.stride = stride

        self.sample_stored_output = None

    def forward(self, inf_data_in, lengths=None):
        if lengths is not None:
            lengths = torch.ceil(lengths.type(torch.double)/self.stride).type(torch.int32)


        if inf_data_in is not None:
            inf_layer_data = self.inf_pre(inf_data_in)

            if self.ar_conn is not None:
                ar_conn = self.ar_conn(inf_layer_data)
                ar_conn = F.pad(ar_conn[:, :, :-1], [1, 0])
            else:
                ar_conn = None
        else:
            inf_layer_data = None
            ar_conn = None

        if self.sublayer is not None:
            next_inf_data = self.inf_post(inf_layer_data)

            if self.inf_batchnorm is not None:
                next_inf_data = self.inf_batchnorm(next_inf_data)

            if lengths is not None:
                max_len = next_inf_data.shape[2]
                length_mask = (torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None])[:, None, :]
                next_inf_data = next_inf_data*length_mask

            gen_data_in, ls = self.sublayer(next_inf_data, lengths)
        else:
            ls = []
            gen_data_in = None

        if not (gen_data_in is None and ar_conn is None):
            gen_layer_data = self.gen_pre(gen_data_in, ar_conn)
        else:
            gen_layer_data = None

        p_data = gen_layer_data
        q_data = torch_ext.cat((inf_layer_data, gen_layer_data), 1)

        if self.latent_q is not None or self.latent_p is not None:
            latent_q_dist = self.latent_q(q_data)

            if p_data is None:
                latent_p_dist = tdist.Normal(torch.zeros_like(latent_q_dist.loc), torch.ones_like(latent_q_dist.loc))
            else:
                latent_p_dist = self.latent_p(p_data)

            latent_q_sample = latent_q_dist.rsample()
            latent_dec = self.latent_dec(latent_q_sample)

            this_ls = [LatentState(latent_q_dist, latent_p_dist, latent_q_sample, lengths)]
        else:
            latent_dec = None
            this_ls = []

        if p_data is not None and self.bypass is not None:
            bypass = self.bypass(p_data)
        else:
            bypass = None

        output_data = self.gen_post(bypass, latent_dec)

        new_ls = ls+this_ls

        if inf_data_in is not None:
            if not np.all(output_data.size()[2:] == inf_data_in.size()[2:]):
                # The way this cuts of additional predictions must be match with how zerodpadding was done in the downsampling step
                if len(inf_data_in.size()) == 4:
                    output_data = output_data[..., :inf_data_in.size(2),
                                                   :inf_data_in.size(3)]
                elif len(inf_data_in.size()) == 3:
                    output_data = output_data[..., :inf_data_in.size(2)]

        if self.gen_batchnorm is not None:
            output_data = self.gen_batchnorm(output_data)

        return output_data, new_ls

    def posterior_parameters(self):
        modules = [self.inf_post, self.inf_pre, self.latent_q]
        params = []

        for module in modules:
            if module is not None:
                params.extend(module.parameters())

        if self.sublayer is not None:
            params.extend(self.sublayer.posterior_parameters())
        return params

    def reset_sample(self):
        self.sample_stored_output = None
        if self.sublayer is not None:
            self.sublayer.reset_sample()

    def sample(self, inf_data_in):

        if self.sample_stored_output is not None:
            output_data = self.sample_stored_output
        else:
            if inf_data_in is not None:
                inf_layer_data = self.inf_pre(inf_data_in)

                if self.ar_conn is not None:
                    ar_conn = self.ar_conn(inf_layer_data)
                    ar_conn = F.pad(ar_conn[:, :, :-1], [1, 0])[..., -1:]
                else:
                    ar_conn = None
            else:
                inf_layer_data = None
                ar_conn = None

            if self.sublayer is not None:
                next_inf_data = self.inf_post(inf_layer_data)

                if self.inf_batchnorm is not None:
                    next_inf_data = self.inf_batchnorm(next_inf_data)

                gen_data_in = self.sublayer.sample(next_inf_data)
            else:
                gen_data_in = None

            if not (gen_data_in is None and ar_conn is None):
                gen_layer_data = self.gen_pre(gen_data_in, ar_conn)
            else:
                gen_layer_data = None

            p_data = gen_layer_data
            q_data = torch_ext.cat((inf_layer_data[..., -1:], gen_layer_data), 1)

            if self.latent_q is not None or self.latent_p is not None:
                latent_q_dist = self.latent_q(q_data)

                if p_data is None:
                    latent_p_dist = tdist.Normal(torch.zeros_like(latent_q_dist.loc), torch.ones_like(latent_q_dist.loc))
                else:
                    latent_p_dist = self.latent_p(p_data)

                # latent_q_dist = torch_ext.combine(latent_q_dist, latent_p_dist)
                latent_sample = latent_p_dist.rsample()
                latent_dec = self.latent_dec(latent_sample)
            else:
                latent_dec = None

            if p_data is not None and self.bypass is not None:
                bypass = self.bypass(p_data)
            else:
                bypass = None

            output_data = self.gen_post(bypass, latent_dec)



        stored_output = output_data[..., 1:]
        self.sample_stored_output = stored_output if stored_output.size()[-1] > 0 else None
        output_data = output_data[..., :1]

        if self.gen_batchnorm is not None:
            output_data = self.gen_batchnorm(output_data)

        return output_data






