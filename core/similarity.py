
import os
import torch
from utils.logger import logger
from core.text_processor import text_similarity
from core.audio_processor import audio_similarity
from core.image_processor import video_similarity
from typing import List, Dict, Union, Optional, TypedDict, Callable


class SimilarityResult(TypedDict):
    link: str
    audio_similarity: float
    image_similarity: float
    text_similarity: float
    overall_similarity: float
    text_meaningful: Optional[bool]
    text_status: Optional[str]


def check_files_exist(files: Union[str, List[str]], check_size: bool = True) -> bool:
    if isinstance(files, str):
        files = [files]
    for file in files:
        try:
            abs_path: str = os.path.abspath(os.path.normpath(file))
            if not os.path.exists(abs_path):
                logger.error(f"文件不存在: {abs_path}")
                return False
            if not os.path.isfile(abs_path):
                logger.error(f"路徑不是文件: {abs_path}")
                return False
            if check_size:
                try:
                    file_size: int = os.path.getsize(abs_path)
                    if file_size == 0:
                        logger.error(f"文件大小為0: {abs_path}")
                        return False
                except Exception as e:
                    logger.error(f"檢查文件大小時出錯 {abs_path}: {str(e)}")
                    return False
        except Exception as e:
            logger.error(f"處理文件路徑時出錯 {file}: {str(e)}")
            return False
    return True


def get_video_id_from_path(file_path: str) -> str:
    basename: str = os.path.basename(file_path)
    if '_frame_' in basename:
        return basename.split('_frame_')[0]
    elif '_transcript' in basename:
        return basename.split('_transcript')[0]
    else:
        return os.path.splitext(basename)[0]


def calculate_overall_similarity(audio1: str, audio2: str, image1: Union[str, List[str]],
                                 image2: Union[str, List[str]], text1: str, text2: str,
                                 video_duration: float, weights: Optional[Dict[str, float]] = None,
                                 emit_cb: Optional[Callable[..., None]] = None, task_id: Optional[str] = None,
                                 link: Optional[str] = None
                                 ) -> Dict[str, Union[float, bool, str, Dict[str, float]]]:
    if weights is None:
        weights = {'audio': 0.3, 'image': 0.4, 'text': 0.3}

    gpu_available: bool = torch.cuda.is_available()
    logger.info("使用 GPU 加速相似度計算" if gpu_available else "使用 CPU 進行相似度計算")
    if emit_cb and task_id:
        emit_cb("progress", task_id=task_id, phase="audio", percent=60, msg="開始計算音頻相似度")

    audio1 = os.path.abspath(os.path.normpath(audio1))
    audio2 = os.path.abspath(os.path.normpath(audio2))

    audio_files_valid: bool = check_files_exist([audio1, audio2], check_size=True)
    if not audio_files_valid:
        logger.error("音頻文件無效，跳過音頻相似度計算")
        a_sim: float = 0.0
    else:
        try:
            logger.info(f"開始計算音頻相似度: {audio1} 和 {audio2}")
            a_sim = audio_similarity(audio1, audio2)
            logger.info(f"音頻相似度: {a_sim:.3f}")
            if emit_cb and task_id:
                emit_cb("progress", task_id=task_id, phase="audio", percent=80, msg=f"音頻相似度: {a_sim:.3f}")
        except Exception as e:
            logger.error(f"計算音頻相似度時出錯: {str(e)}")
            a_sim = 0.0

    # 圖像處理：標準化與驗證
    if emit_cb and task_id:
        emit_cb("progress", task_id=task_id, phase="image", percent=80, msg="開始計算圖像相似度")
    if isinstance(image1, list):
        video_id1: str = get_video_id_from_path(audio1)
        image1_paths: List[str] = [os.path.abspath(os.path.normpath(img)) for img in image1
                                   if get_video_id_from_path(img) == video_id1]
    else:
        image1_paths = [os.path.abspath(os.path.normpath(image1))]

    if isinstance(image2, list):
        video_id2: str = get_video_id_from_path(audio2)
        image2_paths: List[str] = [os.path.abspath(os.path.normpath(img)) for img in image2
                                   if get_video_id_from_path(img) == video_id2]
    else:
        image2_paths = [os.path.abspath(os.path.normpath(image2))]

    frames1_valid: bool = check_files_exist(image1_paths, check_size=True)
    frames2_valid: bool = check_files_exist(image2_paths, check_size=True)
    if not frames1_valid or not frames2_valid:
        logger.error("幀文件無效，跳過圖像相似度計算")
        i_sim: float = 0.0
    else:
        try:
            logger.info(f"開始計算圖像相似度: {len(image1_paths)} 和 {len(image2_paths)} 幀")
            batch_size: int = 64 if gpu_available else 32
            i_sim_result: Dict[str, float] = video_similarity(image1_paths, image2_paths, video_duration, batch_size)
            i_sim = i_sim_result["similarity"]
            logger.info(f"圖像相似度: {i_sim:.3f}")
            if emit_cb and task_id:
                emit_cb("progress", task_id=task_id, phase="image", percent=90, msg=f"圖像相似度: {i_sim:.3f}")
        except Exception as e:
            logger.error(f"計算圖像相似度時出錯: {str(e)}")
            i_sim = 0.0

    # 文本相似度
    if emit_cb and task_id:
        emit_cb("progress", task_id=task_id, phase="text", percent=95, msg="開始計算文本相似度")
    try:
        t_sim: float
        is_meaningful: bool
        reason: str
        t_sim, is_meaningful, reason = text_similarity(text1, text2)
        if not is_meaningful:
            logger.warning(f"文本相似度計算跳過: {reason}")
            text_weight = weights['text']
            total_remaining_weight = weights['audio'] + weights['image']
            if total_remaining_weight > 0:
                audio_weight = weights['audio'] + (text_weight * weights['audio'] / total_remaining_weight)
                image_weight = weights['image'] + (text_weight * weights['image'] / total_remaining_weight)
            else:
                audio_weight = text_weight / 2
                image_weight = text_weight / 2
            weights = {'audio': audio_weight, 'image': image_weight, 'text': 0.0}
            logger.info(f"權重重新分配: 音訊={audio_weight:.3f}, 畫面={image_weight:.3f}, 文本=0.0")
        else:
            weights = weights.copy()
        logger.info(f"文本相似度: {t_sim:.3f} ({reason})")
        if emit_cb and task_id:
            emit_cb("progress", task_id=task_id, phase="text", percent=99, msg=f"文本相似度: {t_sim:.3f}")
    except Exception as e:
        logger.error(f"計算文本相似度時出錯: {str(e)}")
        t_sim = 0.0
        is_meaningful = False
        reason = "錯誤發生"

    overall: float = (weights['audio'] * a_sim + weights['image'] * i_sim + weights['text'] * t_sim)
    logger.info(
        f"整體相似度: {overall:.3f} (音頻={a_sim:.3f}*{weights['audio']:.2f}, 圖像={i_sim:.3f}*{weights['image']:.2f}, 文本={t_sim:.3f}*{weights['text']:.2f})")

    return {
        "audio_similarity": a_sim,
        "image_similarity": i_sim,
        "text_similarity": t_sim,
        "overall_similarity": overall,
        "weights": weights,
        "text_meaningful": is_meaningful,
        "text_status": reason
    }


def display_similarity_results(reference_link: str, comparison_results: List[SimilarityResult]) -> None:
    print("\\n=== 相似度比對結果 ===")
    print(f"參考影片: {reference_link[:60] + '...' if len(reference_link) > 63 else reference_link}\\n")
    print("比對結果:")
    print("-" * 122)
    print(f"{'影片連結':<62} {'音訊相似度':>8} {'畫面相似度':>8} {'內容相似度':>8} {'綜合相似度':>8}")
    print("-" * 122)
    for result in comparison_results:
        link_display = result['link'][:60] + '...' if len(result['link']) > 63 else result['link']
        text_sim_display = f"{result['text_similarity']:>14.3f}"
        if result.get('text_meaningful') is False:
            text_sim_display = f"{result['text_similarity']:>14.3f}*"
        print(f"{link_display:<62} "
              f"{result['audio_similarity']:>14.3f} "
              f"{result['image_similarity']:>14.3f} "
              f"{text_sim_display} "
              f"{result['overall_similarity']:>14.3f}")
    print("-" * 122)
    if any(r.get('text_meaningful') is False for r in comparison_results):
        print("\\n註: * 表示該影片的文本內容被判定為無意義，其文本相似度權重已被重新分配到音訊和畫面相似度")
