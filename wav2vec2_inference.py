import soundfile as sf
import torch
from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2Processor

# Improvements: 
# - convert non 16 khz sample rates
# - inference time log

class Wave2Vec2Inference:
    def __init__(self,model_name, hotwords=[], use_lm_if_possible=True, use_gpu=True):
        # print('we are in wav2vec2 inference class now ')
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        # print('our device is => ',self.device)
        if use_lm_if_possible: 
            # print('will use Auto processor')           
            self.processor = AutoProcessor.from_pretrained(model_name)
        else:
            # print('will use wav2vec2 processor')
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        # print('prepare the model ')
        self.model = AutoModelForCTC.from_pretrained(model_name)
        self.model.to(self.device)
        self.hotwords = hotwords
        self.use_lm_if_possible = use_lm_if_possible
        # print('the model has been prepared')
    def buffer_to_text(self, audio_buffer):
        if len(audio_buffer) == 0:
            return ""

        inputs = self.processor(torch.tensor(audio_buffer), sampling_rate=16_000, return_tensors="pt", padding=True)
        # print('inputs values ==> ',inputs)
        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device),
                                attention_mask=inputs.attention_mask.to(self.device)).logits#[0]            
        # print('logits shape ==> ',logits.shape,logits)
        if hasattr(self.processor, 'decoder') and self.use_lm_if_possible:
            # print('logits as input for decoder ==> ',logits[0].cpu().numpy())
            transcription = self.processor.decode(logits[0].cpu().numpy(),                                      
                                      hotwords=self.hotwords,
                                      #hotword_weight=self.hotword_weight,  
                                      output_word_offsets=True,                                      
                                   )                             
            confidence = transcription.lm_score / len(transcription.text.split(" "))
            transcription = transcription.text       
            # print('transcription when useing LM and decode ==> ',transcription,confidence)
        else:
            predicted_ids = torch.argmax(logits, dim=-1)
            # print('predicted ids after argmax => ', predicted_ids.shape,predicted_ids)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            confidence = self.confidence_score(logits,predicted_ids)
            # print('transcription when useing argmax and batch_decode ==> ',transcription,confidence)
        return transcription, confidence   

    def confidence_score(self, logits, predicted_ids):
        scores = torch.nn.functional.softmax(logits, dim=-1)                                                           
        pred_scores = scores.gather(-1, predicted_ids.unsqueeze(-1))[:, :, 0]
        mask = torch.logical_and(
            predicted_ids.not_equal(self.processor.tokenizer.word_delimiter_token_id), 
            predicted_ids.not_equal(self.processor.tokenizer.pad_token_id))

        character_scores = pred_scores.masked_select(mask)
        total_average = torch.sum(character_scores) / len(character_scores)
        return total_average

    def file_to_text(self, filename):
        audio_input, samplerate = sf.read(filename)
        # print('audio input shape =>',audio_input.shape)
        assert samplerate == 16000
        return self.buffer_to_text(audio_input)
    
if __name__ == "__main__":
    print("Model test")
    model_name = 'facebook/wav2vec2-large-960h-lv60-self'
    # model_name = "oliverguhr/wav2vec2-large-xlsr-53-german-cv9"
    asr = Wave2Vec2Inference(model_name)
    text = asr.file_to_text("test.wav")
    print(text)


