from transformers import (
    SpeechEncoderDecoderModel,
    Wav2Vec2Model,
    AutoFeatureExtractor,
    AutoTokenizer,
    AutoConfig,
    SpeechEncoderDecoderConfig,
)
import torch


encoder_id = "facebook/wav2vec2-base"
decoder_id = "facebook/bart-base"
SAVE_PATH = "../../seq2seq_wav2vec2_bart-base_pretrained_encoder"

# Load pretrained encoder config and modify
config_encoder = AutoConfig.from_pretrained(encoder_id)
config_encoder.add_adapter = True

# Load decoder config without pretrained weights
config_decoder = AutoConfig.from_pretrained(decoder_id)
config_decoder.is_decoder = True
config_decoder.add_cross_attention = True

# Create configuration for the encoder-decoder model
config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(
    config_encoder, config_decoder
)

# Create the model
model_pretrained_encoder = SpeechEncoderDecoderModel(config=config)

# Directly load pretrained Wav2Vec2 encoder weights
pretrained_encoder = Wav2Vec2Model.from_pretrained(encoder_id)

# Manually load state dict, excluding adapter layers
state_dict = pretrained_encoder.state_dict()
model_state_dict = model_pretrained_encoder.encoder.state_dict()

# Remove adapter keys from state_dict
keys_to_load = [k for k in state_dict.keys() if not k.startswith("adapter")]
keys_to_load = [k for k in keys_to_load if k in model_state_dict.keys()]

# Create a new state dict with only the keys we want to load
filtered_state_dict = {k: state_dict[k] for k in keys_to_load}

# Load the filtered state dict
model_pretrained_encoder.encoder.load_state_dict(filtered_state_dict, strict=False)

# Configure model settings
model_pretrained_encoder.config.encoder.feat_proj_dropout = 0.0
model_pretrained_encoder.config.encoder.mask_time_prob = 0.0
model_pretrained_encoder.config.decoder_start_token_id = (
    model_pretrained_encoder.decoder.config.bos_token_id
)
model_pretrained_encoder.config.pad_token_id = (
    model_pretrained_encoder.decoder.config.pad_token_id
)
model_pretrained_encoder.config.eos_token_id = (
    model_pretrained_encoder.decoder.config.eos_token_id
)
model_pretrained_encoder.config.max_length = 128
model_pretrained_encoder.config.encoder.layerdrop = 0.0
model_pretrained_encoder.config.use_cache = False
model_pretrained_encoder.config.processor_class = "Wav2Vec2Processor"

# Check if generation works
_ = model_pretrained_encoder.generate(torch.ones((1, 2000)))

# Save the model
model_pretrained_encoder.save_pretrained(SAVE_PATH)

# Save feature extractor and tokenizer
feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_id)
feature_extractor.save_pretrained(SAVE_PATH)
tokenizer = AutoTokenizer.from_pretrained(decoder_id)
tokenizer.save_pretrained(SAVE_PATH)
