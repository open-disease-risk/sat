"""Initialization of the package"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from transformers import AutoConfig, AutoModel

import sat.models.heads as heads

import sat.models.bert.configuration_bert as sat_bert_config
import sat.models.bert.modeling_bert as sat_bert_model
import sat.models.gpt2.configuration_gpt2 as sat_gpt2_config
import sat.models.gpt2.modeling_gpt2 as sat_gpt2_model

AutoConfig.register(
    sat_bert_config.NumericBertConfig.model_type, sat_bert_config.NumericBertConfig
)
AutoModel.register(sat_bert_config.NumericBertConfig, sat_bert_model.NumericBertModel)

AutoConfig.register(heads.config.SatBertConfig.model_type, heads.config.SatBertConfig)

AutoConfig.register(heads.config.SurvivalConfig.model_type, heads.config.SurvivalConfig)
AutoModel.register(heads.config.SurvivalConfig, heads.SurvivalTaskHead)

AutoConfig.register(
    heads.config.EventClassificationTaskConfig.model_type,
    heads.config.EventClassificationTaskConfig,
)
AutoModel.register(
    heads.config.EventClassificationTaskConfig, heads.EventClassificationTaskHead
)

AutoConfig.register(
    heads.config.EventDurationTaskConfig.model_type,
    heads.config.EventDurationTaskConfig,
)
AutoModel.register(heads.config.EventDurationTaskConfig, heads.EventDurationTaskHead)

AutoConfig.register(heads.config.MTLConfig.model_type, heads.config.MTLConfig)
AutoModel.register(heads.config.MTLConfig, heads.MTLForSurvival)

AutoConfig.register(
    sat_gpt2_config.NumericGPT2Config.model_type, sat_gpt2_config.NumericGPT2Config
)
AutoModel.register(sat_gpt2_config.NumericGPT2Config, sat_gpt2_model.NumericGPT2Model)
