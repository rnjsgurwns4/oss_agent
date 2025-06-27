# Flask API for Emotion Classification
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import pickle
import os
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
import logging
from datetime import datetime
from extract_chat import extract_chat_from_url

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)  # CORS 활성화 (프론트엔드에서 접근 가능하도록)

# 전역 변수로 모델, 토크나이저, 라벨 인코더 저장
model = None
tokenizer = None
label_encoder = None
device = None

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
        return True
        
    except Exception as e:
        logger.error(f"모델 로드 실패: {e}")
        return False

def predict_emotion_api(conversations, last_reply, mbti, gender, age):
    """API용 감정 예측 함수"""
    global model, tokenizer, label_encoder, device
    
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

# API 엔드포인트들

@app.route('/', methods=['GET'])
def health_check():
    """서버 상태 확인 엔드포인트"""
    return jsonify({
        'status': 'OK',
        'message': 'Emotion Classification API is running',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """감정 예측 메인 엔드포인트"""
    try:
        # 모델 로드 확인
        if model is None or tokenizer is None or label_encoder is None:
            return jsonify({
                'error': 'Model not loaded. Please check server logs.',
                'status': 'error'
            }), 500
        
        # 요청 데이터 검증
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'status': 'error'
            }), 400
        
        # 필수 필드 확인
        required_fields = ['url', 'mbti', 'gender', 'age']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}',
                'status': 'error'
            }), 400
        
        url = data['url']
        credential_path = "neural-passkey-463912-c7-9ebc6a6e46f0.json"
        conversations, last_reply = extract_chat_from_url(url, credential_path)
        
        
        
        
        # 데이터 타입 검증
        if not conversations:
            return jsonify({
                'error': 'conversations list cannot be empty',
                'status': 'error'
            }), 400
        
        # 예측 수행
        result = predict_emotion_api(
            conversations = conversations,
            last_reply = last_reply,
            mbti=data['mbti'],
            gender=data['gender'],
            age=data['age']
        )
        
        # 성공 응답
        return jsonify({
            'status': 'success',
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"예측 API 오류: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """배치 예측 엔드포인트 (여러 건 동시 처리)"""
    try:
        # 모델 로드 확인
        if model is None or tokenizer is None or label_encoder is None:
            return jsonify({
                'error': 'Model not loaded. Please check server logs.',
                'status': 'error'
            }), 500
        
        data = request.get_json()
        
        if not data or 'batch' not in data:
            return jsonify({
                'error': 'No batch data provided',
                'status': 'error'
            }), 400
        
        batch_data = data['batch']
        
        if not isinstance(batch_data, list) or not batch_data:
            return jsonify({
                'error': 'batch must be a non-empty list',
                'status': 'error'
            }), 400
        
        results = []
        
        for i, item in enumerate(batch_data):
            try:
                # 필수 필드 확인
                required_fields = ['conversations', 'last_reply', 'mbti', 'gender', 'age']
                missing_fields = [field for field in required_fields if field not in item]
                
                if missing_fields:
                    results.append({
                        'index': i,
                        'status': 'error',
                        'error': f'Missing required fields: {missing_fields}'
                    })
                    continue
                
                # 예측 수행
                result = predict_emotion_api(
                    conversations=item['conversations'],
                    last_reply=item['last_reply'],
                    mbti=item['mbti'],
                    gender=item['gender'],
                    age=item['age']
                )
                
                results.append({
                    'index': i,
                    'status': 'success',
                    'result': result
                })
                
            except Exception as e:
                results.append({
                    'index': i,
                    'status': 'error',
                    'error': str(e)
                })
        
        return jsonify({
            'status': 'completed',
            'results': results,
            'total_processed': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"배치 예측 API 오류: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/info', methods=['GET'])
def model_info():
    """모델 정보 조회 엔드포인트"""
    if label_encoder is None:
        return jsonify({
            'error': 'Model not loaded',
            'status': 'error'
        }), 500
    
    return jsonify({
        'status': 'success',
        'model_info': {
            'available_emotions': list(label_encoder.classes_),
            'num_classes': len(label_encoder.classes_),
            'device': str(device),
            'model_name': 'skt/kobert-base-v1'
        },
        'timestamp': datetime.now().isoformat()
    })

# 에러 핸들러
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error',
        'available_endpoints': [
            'GET /',
            'POST /predict',
            'POST /predict/batch',
            'GET /info'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500

if __name__ == '__main__':
    # 서버 시작 전 모델 로드
    logger.info("Flask 서버 시작 준비 중...")
    
    # 모델 파일 경로 설정 (필요에 따라 수정)
    model_directory = "./"  # 현재 디렉토리
    # model_directory = "/content/drive/MyDrive/"  # 구글 코랩용
    
    if load_model_files(model_directory):
        logger.info("모델 로드 성공! 서버를 시작합니다.")
        # 서버 실행
        app.run(
            host='0.0.0.0',  # 외부 접근 허용
            port=5000,       # 포트 번호
            debug=False,     # 프로덕션에서는 False
            threaded=True    # 멀티스레딩 지원
        )
    else:
        logger.error("모델 로드 실패! 서버를 시작할 수 없습니다.")
        logger.error("다음 파일들이 올바른 경로에 있는지 확인하세요:")
        logger.error("  - emotion_classifier.pth")
        logger.error("  - label_encoder.pkl")