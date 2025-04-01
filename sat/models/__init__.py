"""Initialization of the package"""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from transformers import AutoConfig, AutoModel

import sat.models.bert.configuration_bert as sat_bert_config
import sat.models.bert.modeling_bert as sat_bert_model
import sat.models.gpt2.configuration_gpt2 as sat_gpt2_config
import sat.models.gpt2.modeling_gpt2 as sat_gpt2_model
import sat.models.heads as heads

AutoConfig.register(
    sat_bert_config.NumericBertConfig.model_type, sat_bert_config.NumericBertConfig
)
AutoModel.register(sat_bert_config.NumericBertConfig, sat_bert_model.NumericBertModel)

AutoConfig.register(heads.SatBertConfig.model_type, heads.SatBertConfig)

AutoConfig.register(heads.SurvivalConfig.model_type, heads.SurvivalConfig)
AutoModel.register(heads.SurvivalConfig, heads.SurvivalTaskHead)

# Register MENSA
AutoConfig.register(heads.MENSAConfig.model_type, heads.MENSAConfig)
AutoModel.register(heads.MENSAConfig, heads.MENSATaskHead)

AutoConfig.register(
    heads.EventClassificationTaskConfig.model_type,
    heads.EventClassificationTaskConfig,
)
AutoModel.register(
    heads.EventClassificationTaskConfig, heads.EventClassificationTaskHead
)

AutoConfig.register(
    heads.EventDurationTaskConfig.model_type,
    heads.EventDurationTaskConfig,
)
AutoModel.register(heads.EventDurationTaskConfig, heads.EventDurationTaskHead)

AutoConfig.register(heads.MTLConfig.model_type, heads.MTLConfig)
AutoModel.register(heads.MTLConfig, heads.MTLForSurvival)

AutoConfig.register(
    sat_gpt2_config.NumericGPT2Config.model_type, sat_gpt2_config.NumericGPT2Config
)
AutoModel.register(sat_gpt2_config.NumericGPT2Config, sat_gpt2_model.NumericGPT2Model)
