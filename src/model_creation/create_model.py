from transformers import SpeechEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer
import torch


encoder_id = "facebook/wav2vec2-base"
decoder_id = "facebook/bart-base"
SAVE_PATH = "../../seq2seq_wav2vec2_bart-base"

model = SpeechEncoderDecoderModel.from_encoder_decoder_pretrained(
    encoder_id, decoder_id, encoder_add_adapter=True
)
model.config.encoder.feat_proj_dropout = 0.0
model.config.encoder.mask_time_prob = 0.0
model.config.decoder_start_token_id = model.decoder.config.bos_token_id
model.config.pad_token_id = model.decoder.config.pad_token_id
model.config.eos_token_id = model.decoder.config.eos_token_id
model.config.max_length = 128
model.config.encoder.layerdrop = 0.0
model.config.use_cache = False
model.config.processor_class = "Wav2Vec2Processor"

# check if forward pass works
dummy_input = torch.ones((1, 2000))
outputs = model.forward(dummy_input)

model.save_pretrained(SAVE_PATH)

feature_etxractor = AutoFeatureExtractor.from_pretrained(encoder_id)
feature_etxractor.save_pretrained(SAVE_PATH)
tokenizer = AutoTokenizer.from_pretrained(decoder_id)
tokenizer.save_pretrained(SAVE_PATH)
