
import torch

import logging



# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_emotion_api(model, tokenizer, label_encoder, device, conversations, last_reply, mbti, gender, age):
    """API용 감정 예측 함수"""
    #global model, tokenizer, label_encoder, device
    
    try:
        model.eval()
        
        # 텍스트 전처리
        conversations_text = " [SEP] ".join(conversations)
        full_text = f"대화: {conversations_text} [SEP] 마지막답변: {last_reply} [SEP] MBTI: {mbti} [SEP] 성별: {gender} [SEP] 나이: {age}"
        
        # 토크나이징
        encoding = tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        # 예측
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask)
            predictions = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1)
            
            emotion = label_encoder.inverse_transform(predicted_class.cpu().numpy())[0]
            confidence = predictions.max().item()
            
            # 모든 감정의 확률 계산 (선택사항)
            all_probabilities = {}
            for i, emotion_name in enumerate(label_encoder.classes_):
                all_probabilities[emotion_name] = float(predictions[0][i].cpu().numpy())
            
            return {
                'emotion': emotion,
                'confidence': float(confidence),
                'all_probabilities': all_probabilities
            }
            
    except Exception as e:
        logger.error(f"예측 중 오류 발생: {e}")
        raise e