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
SAVE_PATH = "../../seq2seq_wav2vec2_bart-base_pretrained_decoder"

# Load pretrained encoder config and modify
config_encoder = AutoConfig.from_pretrained(encoder_id)
config_encoder.add_adapter = True

# Load decoder config with pretrained weights
config_decoder = AutoConfig.from_pretrained(decoder_id)
config_decoder.is_decoder = True
config_decoder.add_cross_attention = True

# Create configuration for the encoder-decoder model
config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(
    config_encoder, config_decoder
)

# Create the model with randomly initialized weights
model_pretrained_decoder = SpeechEncoderDecoderModel(config=config)

# Directly load pretrained BART decoder weights
pretrained_decoder = AutoConfig.from_pretrained(decoder_id)

# Manually load decoder state dict
decoder_state_dict = model_pretrained_decoder.decoder.state_dict()
pretrained_decoder_model = model_pretrained_decoder.decoder
pretrained_decoder_state_dict = pretrained_decoder_model.from_pretrained(
    decoder_id
).state_dict()

# Load the decoder state dict
model_pretrained_decoder.decoder.load_state_dict(pretrained_decoder_state_dict)

# Configure model settings
model_pretrained_decoder.config.encoder.feat_proj_dropout = 0.0
model_pretrained_decoder.config.encoder.mask_time_prob = 0.0
model_pretrained_decoder.config.decoder_start_token_id = (
    model_pretrained_decoder.decoder.config.bos_token_id
)
model_pretrained_decoder.config.pad_token_id = (
    model_pretrained_decoder.decoder.config.pad_token_id
)
model_pretrained_decoder.config.eos_token_id = (
    model_pretrained_decoder.decoder.config.eos_token_id
)
model_pretrained_decoder.config.max_length = 128
model_pretrained_decoder.config.encoder.layerdrop = 0.0
model_pretrained_decoder.config.use_cache = False
model_pretrained_decoder.config.processor_class = "Wav2Vec2Processor"

# Check if generation works
_ = model_pretrained_decoder.generate(torch.ones((1, 2000)))

# Save the model
model_pretrained_decoder.save_pretrained(SAVE_PATH)

# Save feature extractor and tokenizer
feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_id)
feature_extractor.save_pretrained(SAVE_PATH)
tokenizer = AutoTokenizer.from_pretrained(decoder_id)
tokenizer.save_pretrained(SAVE_PATH)
