import requests
import io
from google.cloud import vision
import os
import cv2
import numpy as np
import re

def extract_chat_from_url(image_url: str, credential_path: str) -> list[str]:
    
    
    # 1. 환경변수 설정
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

    # 2. 이미지 다운로드 및 열기
    response = requests.get(image_url)
    image_bytes = io.BytesIO(response.content)
    image_array = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image_np=image

    
    # 3. 이미지 위아래 자르기(ui 등 쓸모없는 글 삭제)
    def crop_chat_region(image_np: np.ndarray, top_ratio=0.10, bottom_ratio=0.10) -> np.ndarray:
        height = image_np.shape[0]
        top = int(height * top_ratio)
        bottom = int(height * (1 - bottom_ratio))
        cropped = image_np[top:bottom, :]
        return cropped
    
    # 이미지 자름
    image_np = crop_chat_region(image_np)

    # 4. Google Vision OCR 요청
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=cv2.imencode(".png", image_np)[1].tobytes())
    response = client.text_detection(image=image)
    annotations = response.text_annotations
    
    
    
    # 5. 위치와 색깔로 화자 구분
    def classify_speaker_by_position_and_color(text_annotations, image_np):
        messages = []
        left_colors = []
        right_colors = []

        # Step 1: 말풍선 텍스트 중심과 배경 색 추출
        for text in text_annotations[1:]:
            desc = text.description
            box = text.bounding_poly.vertices
            xs = [v.x for v in box if v.x is not None]
            ys = [v.y for v in box if v.y is not None]
            x_center = int(sum(xs) / len(xs))
            y_center = int(sum(ys) / len(ys))

            # 주변 색상 추출 (중심점 기준 5x5 픽셀)
            x_min = max(0, x_center - 2)
            x_max = min(image_np.shape[1], x_center + 2)
            y_min = max(0, y_center - 2)
            y_max = min(image_np.shape[0], y_center + 2)
            crop = image_np[y_min:y_max, x_min:x_max]
            avg_color = crop.mean(axis=(0, 1))  # BGR 평균

            if x_center < image_np.shape[1] / 2:
                left_colors.append(avg_color)
            else:
                right_colors.append(avg_color)

            messages.append({
                "text": desc,
                "x": x_center,
                "color": avg_color
                })

        # Step 2: 왼쪽/오른쪽 평균 색상 차이 계산
        import numpy as np
        left_mean = np.mean(left_colors, axis=0) if left_colors else np.array([0, 0, 0])
        right_mean = np.mean(right_colors, axis=0) if right_colors else np.array([0, 0, 0])
        color_diff = np.linalg.norm(left_mean - right_mean)


        # Step 3: 말풍선별 화자 결정
        final = []
        for m in messages:
            # 중심 좌표 + 색 차이 기준 분류
            if color_diff > 20:
                speaker = "상대방" if np.linalg.norm(m["color"] - left_mean) < np.linalg.norm(m["color"] - right_mean) else "나"
            else:
                # 색 차이 크지 않으면 위치로만 판단
                speaker = "상대방" if m["x"] < image_np.shape[1] / 2 else "나"

            final.append((speaker, m["text"]))

        return final
    
    
    
    
    # 6. 화자 병합
    def merge_messages(messages):
        merged = []
        current_speaker = None
        current_text = ""
        
        # ['나', '나', '나', '상대방' 상대방'] -> ['나', '상대방'] 이런 식으로 병합
        for speaker, text in messages:
            if speaker != current_speaker:
                if current_speaker is not None:
                    merged.append(f"{current_speaker}")
                current_speaker = speaker
                current_text = text
            else:
                current_text += " " + text

        if current_speaker:
            
            merged.append(f"{current_speaker}")
        return merged




    # 7. 대화 추출
    def split_messages_by_time_block(full_text: str) -> list[str]:
        lines = full_text.strip().split('\n')
        
        #오전/오후 0:00로 대화를 쪼갬
        time_pattern = re.compile(r'(오전|오후)\s?\d{1,2}:\d{2}')

        blocks = []
        current_block = []
        
        for line in lines:
            if time_pattern.fullmatch(line.strip()):
                # 시간 텍스트가 나오면 이전 블록 저장
                if current_block:
                    blocks.append(current_block)
                    current_block = []
            else:
                # 시간 아닌 텍스트면 누적
                if line.strip() not in ["", "|메시지 입력", "전송"]:
                    current_block.append(line.strip())

        # 마지막 블록 추가
        if current_block:
            blocks.append(current_block)

        # 블락에 메시지 담기
        result = []
        for block in blocks:
            if not block:
                continue
            message = " ".join(block[0:]) if len(block) > 1 else ""
            result.append(f"{message.strip()}")
        
        # 마지막 항목이 빈 문자열이면 제거
        if result and result[-1].strip() == "":
            result.pop()

        return result


    
    # 8. 필터링 및 병합
    def filtered_messages(speak_merged, messages):
        filtered = []
        last_reply = None
        last_reply_index = -1

        for idx, (speaker, content) in enumerate(zip(speak_merged, messages)):
            if not content.strip():
                continue

            # 상대방: 이후에 나오는 첫 단어(이름) 제거
            if speaker.strip() == "상대방":
                words = content.strip().split()
                if len(words) > 1:
                    content = " ".join(words[1:])
                else:
                    content = ""

            formatted = f"{speaker.strip()}: {content.strip()}"

            # 상대방의 마지막 말 위치 기억
            if speaker.strip() == "상대방":
                last_reply = formatted
                last_reply_index = len(filtered)  # 이 위치까지만 유지됨

            filtered.append(formatted)

        # 상대방 마지막 말 이후의 "나:" 채팅 제거
        if last_reply_index != -1:
            filtered = filtered[:last_reply_index]  # 마지막 상대방 전까지 유지


        return filtered, last_reply
    
    
    #함수 실행
    messages = split_messages_by_time_block(annotations[0].description)
    speakers = classify_speaker_by_position_and_color(annotations, image_np)
    speak_merged = merge_messages(speakers)


    """
    테스트용
    speak_merged = ['나', '상대방', '나', '상대방','나','상대방']
    messages = ['agentica 잘 작동 하나 확인했는 이거 진짜 왜 되는건지 모르겠음', '손지운 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ github 파헤쳐봐야 알듯', '함수 자동으로 불러오는 거 이거 진짜 신기함 너도 해보셈', '손지운 오', '시발년아', '손지운 뭐']
    """
    
    filtered, last_reply = filtered_messages(speak_merged, messages)

    return filtered, last_reply

"""
#main
image_url = "https://app-culture-bucket.s3.ap-southeast-2.amazonaws.com/%ED%99%94%EB%A9%B4+%EC%BA%A1%EC%B2%98+2025-06-24+203211.png"
credential_path = "neural-passkey-463912-c7-9ebc6a6e46f0.json"

#filtered: 대화들, last_reply: 상대방의 마지막 대화
filtered, last_reply = extract_chat_from_url(image_url, credential_path)

    
print(filtered)
print(last_reply)
"""
    

