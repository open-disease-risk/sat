"""Initialization of the package"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from transformers import AutoConfig, AutoModel

import sat.models.tasks.config as config
import sat.models.tasks.heads as heads

import sat.models.bert.configuration_bert as sat_bert_config
import sat.models.bert.modeling_bert as sat_bert_model
import sat.models.gpt2.configuration_gpt2 as sat_gpt2_config
import sat.models.gpt2.modeling_gpt2 as sat_gpt2_model

AutoConfig.register(
    sat_bert_config.NumericBertConfig.model_type, sat_bert_config.NumericBertConfig
)
AutoModel.register(sat_bert_config.NumericBertConfig, sat_bert_model.NumericBertModel)

AutoConfig.register(config.SatBertConfig.model_type, config.SatBertConfig)

AutoConfig.register(config.SurvivalConfig.model_type, config.SurvivalConfig)
AutoModel.register(config.SurvivalConfig, heads.SurvivalTaskHead)

AutoConfig.register(
    config.EventClassificationTaskConfig.model_type,
    config.EventClassificationTaskConfig,
)
AutoModel.register(
    config.EventClassificationTaskConfig, heads.EventClassificationTaskHead
)

AutoConfig.register(
    config.EventDurationTaskConfig.model_type, config.EventDurationTaskConfig
)
AutoModel.register(config.EventDurationTaskConfig, heads.EventDurationTaskHead)

AutoConfig.register(config.MTLConfig.model_type, config.MTLConfig)
AutoModel.register(config.MTLConfig, heads.MTLForSurvival)

AutoConfig.register(
    sat_gpt2_config.NumericGPT2Config.model_type, sat_gpt2_config.NumericGPT2Config
)
AutoModel.register(sat_gpt2_config.NumericGPT2Config, sat_gpt2_model.NumericGPT2Model)
