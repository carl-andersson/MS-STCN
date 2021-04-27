
from model.DistModules import *
from model.torch_ext_causal_conv import *
from model.VariationalLayers.AutoregressiveLadderLayer import AutoregressiveLadderLayer, AutoregressiveLayerBuilder
from model.VariationalLayers.LadderLayer import LadderLayer, LayerBuilder

from model.torch_wrappers import ModuleWrapper







class MultiscaleVAE(nn.Module):
    def __init__(self, input_size, output_size, ar_layers_options, default_ar_layer, fc_layers_options, default_fc_layer):
        super().__init__()
        first_builder = None
        last_builder = None

        for layer in ar_layers_options:
            if first_builder is None:
                first_builder = AutoregressiveLayerBuilder({**default_ar_layer, **layer},
                                                           input_size=input_size, output_size=output_size)
                last_builder = first_builder
            else:
                last_builder = AutoregressiveLayerBuilder({**default_ar_layer, **layer}, prev_layer=last_builder)

        for layer in fc_layers_options:
            if first_builder is None:
                first_builder = LayerBuilder({**default_fc_layer, **layer},
                                             input_size=input_size, output_size=output_size)
                last_builder = first_builder
            else:
                last_builder = LayerBuilder({**default_fc_layer, **layer}, prev_layer=last_builder)

        self._output_size = first_builder.output_size
        self.first_ladder_layer = first_builder.build()

    @property
    def output_size(self):
        return self._output_size

    def forward(self, x, lengths=None):
        return self.first_ladder_layer(x, lengths=lengths)

    def posterior_parameters(self):
        if self.first_ladder_layer is not None:
            return self.first_ladder_layer.posterior_parameters()

    def apply_batchnorm(self):
        c_layer = self.first_ladder_layer
        while c_layer is not None:
            if c_layer.inf_post is not None:
                c_layer.inf_batchnorm = nn.BatchNorm1d(c_layer.inf_post.out_channels, affine=False)
            else:
                c_layer.inf_batchnorm = None
            c_layer.gen_batchnorm = nn.BatchNorm1d(c_layer.gen_post.out_channels, affine=False)
            c_layer = c_layer.sublayer

    def sample(self, cond):
        return self.first_ladder_layer.sample(cond)

    def reset_sample(self):
        self.first_ladder_layer.reset_sample()
