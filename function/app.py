# Flask API for Emotion Classification
from flask import Flask, request, jsonify
from flask_cors import CORS

import logging
from datetime import datetime
from extract_chat import extract_chat_from_url
from feature_emotion import predict_emotion_api
from model_loader import load_model_files

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
            model = model,
            tokenizer = tokenizer, 
            label_encoder = label_encoder, 
            device = device,
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
    
    model, tokenizer, label_encoder, device = load_model_files("./")
    if model is not None:
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