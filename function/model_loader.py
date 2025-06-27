
import torch
import torch.nn as nn
import pickle
import os
from transformers import AutoTokenizer, AutoModel

import logging


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 모델 클래스 정의
class EmotionClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout_rate=0.3):
        super(EmotionClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        dropped_output = self.dropout(pooled_output)
        logits = self.classifier(dropped_output)
        return logits

def load_model_files(model_dir="./"):
    """모델 파일들을 로드하는 함수"""
    global model, tokenizer, label_encoder, device
    
    logger.info("모델 로딩 시작...")
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"사용 중인 디바이스: {device}")
    
    model_name = "skt/kobert-base-v1"
    
    try:
        # 1. 라벨 인코더 로드
        logger.info("라벨 인코더 로드...")
        with open(f"{model_dir}label_encoder.pkl", 'rb') as f:
            label_encoder = pickle.load(f)
        logger.info(f"감정 라벨: {list(label_encoder.classes_)}")

        # 2. 토크나이저 로드
        logger.info("토크나이저 로드...")
        tokenizer_path = f"{model_dir}saved_tokenizer"
        
        if os.path.exists(tokenizer_path):
            logger.info("저장된 토크나이저 발견, 로드 중...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            logger.info("인터넷에서 토크나이저 다운로드 중...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # 토크나이저를 로컬에 저장 (선택사항)
            try:
                logger.info("토크나이저를 로컬에 저장 중...")
                tokenizer.save_pretrained(tokenizer_path)
                logger.info("토크나이저 저장 완료!")
            except Exception as save_error:
                logger.warning(f"토크나이저 저장 실패: {save_error}")

        # 3. 모델 로드
        logger.info("모델 로드...")
        model = EmotionClassifier(model_name, len(label_encoder.classes_))
        model.load_state_dict(torch.load(f"{model_dir}emotion_classifier.pth", map_location=device))
        model.to(device)
        model.eval()
        
        logger.info("모든 모델 파일 로드 완료!")
        return model, tokenizer, label_encoder, device
        
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        return None, None, None, None