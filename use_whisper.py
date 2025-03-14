import whisper
import argparse
import os
import time

def transcribe_audio(audio_path, model_name="base", verbose=True, word_timestamps=True):
    """
    오디오 파일을 트랜스크립션합니다.
    
    Args:
        audio_path: 오디오 파일 경로
        model_name: 사용할 Whisper 모델 이름 (tiny, base, small, medium, large, turbo)
        verbose: 상세 출력 여부
    
    Returns:
        트랜스크립션 결과 딕셔너리
    """
    start_time = time.time()
    
    if verbose:
        print(f"모델 '{model_name}' 로딩 중...")
    
    # 모델 로드
    model = whisper.load_model(model_name)
    
    if verbose:
        print(f"모델 로딩 완료: {time.time() - start_time:.2f}초")
        print(f"'{audio_path}' 파일 트랜스크립션 시작...")
    
    # 오디오 로드 및 트랜스크립션
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    
    # 스펙트로그램 생성 (detected_lang은 log_mel_spectrogram에 직접 영향 주지는 않음)
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    
    # 언어 감지
    _, probs = model.detect_language(mel)
    detected_lang = max(probs, key=probs.get)
    
    if verbose:
        print(f"감지된 언어: {detected_lang} (확률: {probs[detected_lang]:.2f})")
    
    # 오디오 디코딩
    options = whisper.DecodingOptions(language=detected_lang, fp16=False)
    _ = whisper.decode(model, mel, options)
    
    # 전체 트랜스크립션 (더 긴 오디오 파일을 위해)
    # => word_timestamps 인자 추가
    full_result = model.transcribe(
        audio_path,
        language=detected_lang,
        word_timestamps=True
    )
    
    if verbose:
        print(f"트랜스크립션 완료: {time.time() - start_time:.2f}초")
    
    return full_result

def format_time(seconds):
    """
    초 단위 시간을 '00:00:00.000' 형식으로 변환
    """
    milliseconds = int((seconds - int(seconds)) * 1000)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"

def print_word_timestamps(full_result):
    """
    segments 단위로 순회하며, 단어 단위 start/end 시간 정보를 출력.
    """
    segments = full_result.get("segments", [])
    for seg in segments:
        # seg_start = seg["start"]
        # seg_end   = seg["end"]
        seg_text  = seg["text"]
        # print(f"[Segment] {seg_start:.3f}~{seg_end:.3f} \"{seg_text}\"")
        words = seg.get("words", [])
        if words and len(words) > 0:
            start_time = format_time(words[0]["start"])
            end_time = format_time(words[-1]["end"])
            print(f"{start_time} --> {end_time}\n [segment]: {seg_text}")

    # print()


def save_transcription(result, output_file=None, audio_path=None):
    """
    트랜스크립션 결과를 파일에 저장합니다.
    
    Args:
        result: 트랜스크립션 결과 딕셔너리
        output_file: 출력 파일 경로 (None인 경우 오디오 파일명 + .txt)
        audio_path: 원본 오디오 파일 경로
    """
    
    # 결과 파일을 저장할 폴더
    result_dir = "result"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)


    if output_file is None and audio_path is not None:
        # 오디오 파일 이름에서 확장자 제거하고 .txt 추가
        base_name = os.path.basename(audio_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_file = f"{result_dir}/{name_without_ext}.txt"
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result["text"])
        print(f"트랜스크립션 결과가 '{output_file}'에 저장되었습니다.")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Whisper 오디오 트랜스크립션 도구")
    parser.add_argument("audio_file", nargs="?", default=None, help="트랜스크립션할 오디오 파일 경로")
    parser.add_argument("--folder", "-f", default=None, help="트랜스크립션할 오디오 파일이 있는 폴더 경로 (파일명을 입력하지 않으면 기본값: ./target 폴더 사용)")
    parser.add_argument("--model", "-m", default="medium", 
                        choices=["tiny", "base", "small", "medium", "large", "turbo"],
                        help="사용할 Whisper 모델")
    parser.add_argument("--output", "-o", help="출력 파일 경로 (기본값: 오디오 파일명.txt)")
    parser.add_argument("--quiet", "-q", action="store_true", help="상세 출력 비활성화")
    
    args = parser.parse_args()

    # 폴더가 지정된 경우 해당 폴더 내 파일 검색
    target_dir = args.folder if args.folder else "./target"

    # 파일이 지정되지 않은 경우 ./target 폴더에서 자동 검색
    if args.audio_file is None:
        if not os.path.exists(target_dir):
            print(f"오류: '{target_dir}' 폴더를 찾을 수 없습니다.")
            return
        
        audio_files = [f for f in os.listdir(target_dir) if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a'))]
        
        if not audio_files:
            print(f"오류: '{target_dir}' 폴더에 음원 파일이 없습니다.")
            return
        
        args.audio_file = os.path.join(target_dir, audio_files[0])
        print(f"자동 선택된 오디오 파일: {args.audio_file}")
    else:
        if not os.path.exists(args.audio_file):
            print(f"오류: '{args.audio_file}' 파일을 찾을 수 없습니다")
            return
    
    # 트랜스크립션 실행
    result = transcribe_audio(args.audio_file, args.model, verbose=not args.quiet, word_timestamps=True)
    
    # 결과 저장
    save_transcription(result, args.output, args.audio_file)
    
    # 단어 타임스탬프 출력 (optional)
    print("\n[단어 단위 타임스탬프 정보]")
    print_word_timestamps(result)
    
    # 화면에 출력
    print("\n트랜스크립션 결과:")
    print(result["text"])

if __name__ == "__main__":
    main()