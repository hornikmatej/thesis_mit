from transformers import (
    SpeechEncoderDecoderModel,
    AutoFeatureExtractor,
    AutoTokenizer,
    AutoConfig,
    SpeechEncoderDecoderConfig,
)
import torch


encoder_id = "facebook/wav2vec2-base"
decoder_id = "facebook/bart-base"
SAVE_PATH = "../../seq2seq_wav2vec2_bart-base_scratch"


# Load model without pretrained weights
config_encoder = AutoConfig.from_pretrained(encoder_id)
config_encoder.add_adapter = True
config_decoder = AutoConfig.from_pretrained(decoder_id)
config_decoder.is_decoder = True
config_decoder.add_cross_attention = True

config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(
    config_encoder, config_decoder
)
model_scratch = SpeechEncoderDecoderModel(config=config)

model_scratch.config.encoder.feat_proj_dropout = 0.0
model_scratch.config.encoder.mask_time_prob = 0.0
model_scratch.config.decoder_start_token_id = model_scratch.decoder.config.bos_token_id
model_scratch.config.pad_token_id = model_scratch.decoder.config.pad_token_id
model_scratch.config.eos_token_id = model_scratch.decoder.config.eos_token_id
model_scratch.config.max_length = 128
model_scratch.config.encoder.layerdrop = 0.0
model_scratch.config.use_cache = False
model_scratch.config.processor_class = "Wav2Vec2Processor"

# check if generation works
_ = model_scratch.generate(torch.ones((1, 2000)))

model_scratch.save_pretrained(SAVE_PATH)

feature_etxractor = AutoFeatureExtractor.from_pretrained(encoder_id)
feature_etxractor.save_pretrained(SAVE_PATH)
tokenizer = AutoTokenizer.from_pretrained(decoder_id)
tokenizer.save_pretrained(SAVE_PATH)
