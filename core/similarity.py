# ======================== 📦 模組與依賴 ========================
import os
import torch
import numpy as np
from utils.logger import logger
from typing import List, Dict, Union
from core.text_processor import text_similarity
from concurrent.futures import ThreadPoolExecutor
from core.audio_processor import audio_similarity
from core.image_processor import video_similarity

# ======================== 📂 檔案驗證與前處理 ========================
def check_files_exist(files: Union[str, List[str]], check_size: bool = True) -> bool:
    """
    檢查文件是否存在且有效
    
    參數:
        files: 單個文件路徑或文件路徑列表（支持相對路徑和絕對路徑）
        check_size: 是否檢查文件大小
    
    返回:
        所有文件都存在且有效返回 True，否則返回 False
    """
    if isinstance(files, str):
        files = [files]
        
    for file in files:
        try:
            # 標準化路徑
            abs_path = os.path.abspath(os.path.normpath(file))
            
            # 檢查文件是否存在
            if not os.path.exists(abs_path):
                logger.error(f"文件不存在: {abs_path}")
                return False
                
            # 檢查是否為文件（不是目錄）
            if not os.path.isfile(abs_path):
                logger.error(f"路徑不是文件: {abs_path}")
                return False
                
            # 檢查文件大小
            if check_size:
                try:
                    file_size = os.path.getsize(abs_path)
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
    """從文件路徑中提取影片ID（包含播放清單索引）"""
    basename = os.path.basename(file_path)
    # 處理不同類型的文件名
    if '_frame_' in basename:  # 幀文件
        return basename.split('_frame_')[0]
    elif '_transcript' in basename:  # 轉錄文件
        return basename.split('_transcript')[0]
    else:  # 影片或音頻文件
        return os.path.splitext(basename)[0]

# ======================== 🎯 相似度計算主流程 ========================
def calculate_overall_similarity(audio1: str, audio2: str,
                               image1: Union[str, List[str]], image2: Union[str, List[str]],
                               text1: str, text2: str,
                               video_duration: float,
                               weights: dict = None) -> Dict[str, float]:
    """
    計算整體相似度
    
    參數:
        audio1, audio2: 音訊文件路徑
        image1, image2: 圖像文件路徑或幀列表
        text1, text2: 文本內容
        video_duration: 視頻時長（秒）
        weights: 各部分的權重
    
    返回:
        包含各部分相似度和整體相似度的字典
    """
    if weights is None:
        weights = {'audio': 0.3, 'image': 0.4, 'text': 0.3}
    
    # 檢查 GPU 可用性
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        logger.info("使用 GPU 加速相似度計算")
    else:
        logger.info("使用 CPU 進行相似度計算")
    
    # 標準化音頻文件路徑
    audio1 = os.path.abspath(os.path.normpath(audio1))
    audio2 = os.path.abspath(os.path.normpath(audio2))
    
    # 檢查音頻文件
    audio_files_valid = check_files_exist([audio1, audio2], check_size=True)
    if not audio_files_valid:
        logger.error(f"音頻文件無效，跳過音頻相似度計算")
        logger.error(f"音頻文件1: {audio1}")
        logger.error(f"音頻文件2: {audio2}")
        a_sim = 0
    else:
        # 計算音訊相似度
        try:
            logger.info(f"開始計算音頻相似度: {audio1} 和 {audio2}")
            a_sim = audio_similarity(audio1, audio2)
            logger.info(f"音頻相似度: {a_sim:.3f}")
        except Exception as e:
            logger.error(f"計算音頻相似度時出錯: {str(e)}")
            logger.error(f"音頻文件1: {audio1}")
            logger.error(f"音頻文件2: {audio2}")
            a_sim = 0
    
    # 標準化圖像文件路徑並確保它們屬於正確的影片
    if isinstance(image1, list):
        video_id1 = get_video_id_from_path(audio1)
        image1 = [os.path.abspath(os.path.normpath(img)) for img in image1 
                 if get_video_id_from_path(img) == video_id1]
    else:
        image1 = [os.path.abspath(os.path.normpath(image1))]
        
    if isinstance(image2, list):
        video_id2 = get_video_id_from_path(audio2)
        image2 = [os.path.abspath(os.path.normpath(img)) for img in image2 
                 if get_video_id_from_path(img) == video_id2]
    else:
        image2 = [os.path.abspath(os.path.normpath(image2))]
    
    # 計算圖像相似度
    # 檢查幀文件
    frames1_valid = check_files_exist(image1, check_size=True)
    frames2_valid = check_files_exist(image2, check_size=True)
    if not frames1_valid or not frames2_valid:
        logger.error("幀文件無效，跳過圖像相似度計算")
        i_sim = 0
    else:
        try:
            # 使用 pHash 進行視頻相似度比對
            logger.info(f"開始計算圖像相似度: {len(image1)} 和 {len(image2)} 幀")
            batch_size = 64 if gpu_available else 32
            i_sim_result = video_similarity(image1, image2, video_duration, batch_size)
            i_sim = i_sim_result["similarity"]
            logger.info(f"圖像相似度: {i_sim:.3f}")
        except Exception as e:
            logger.error(f"計算圖像相似度時出錯: {str(e)}")
            i_sim = 0
    
    # 計算文本相似度
    try:
        logger.info("開始計算文本相似度")
        t_sim, is_meaningful, reason = text_similarity(text1, text2)
        
        # 根據文本相似度的結果動態調整權重
        if not is_meaningful:
            logger.warning(f"文本相似度計算跳過: {reason}")
            # 如果文本無意義，將權重重新分配給音頻和圖像
            audio_weight = weights['audio'] / (weights['audio'] + weights['image'])
            image_weight = weights['image'] / (weights['audio'] + weights['image'])
            weights = {
                'audio': audio_weight,
                'image': image_weight,
                'text': 0  # 文本權重設為0
            }
        else:
            # 如果文本有意義，使用原始權重
            weights = weights.copy()
            
        logger.info(f"文本相似度: {t_sim:.3f} ({reason})")
    except Exception as e:
        logger.error(f"計算文本相似度時出錯: {str(e)}")
        t_sim = 0
    
    # 計算加權總分
    overall = weights['audio']*a_sim + weights['image']*i_sim + weights['text']*t_sim
    logger.info(f"整體相似度: {overall:.3f} (音頻={a_sim:.3f}*{weights['audio']:.2f}, "
               f"圖像={i_sim:.3f}*{weights['image']:.2f}, "
               f"文本={t_sim:.3f}*{weights['text']:.2f})")
    
    # 返回所有相似度結果和權重信息
    return {
        "audio_similarity": a_sim,
        "image_similarity": i_sim,
        "text_similarity": t_sim,
        "overall_similarity": overall,
        "weights": weights,
        "text_meaningful": is_meaningful,
        "text_status": reason
    }

# ======================== 📊 結果輸出與顯示 ========================
def display_similarity_results(reference_link: str, comparison_results: List[Dict[str, Union[str, float]]]):
    """
    顯示相似度比對結果
    
    參數:
        reference_link: 參考影片的連結
        comparison_results: 包含比對結果的列表，每個結果包含 link 和相似度分數
    """
    print("\n=== 相似度比對結果 ===")
    print(f"參考影片: {reference_link[:60] + '...' if len(reference_link) > 63 else reference_link}\n")
    print("比對結果:")
    print("-" * 120)
    print(f"{'影片連結':<62} {'音訊相似度':>8} {'畫面相似度':>8} {'內容相似度':>8} {'綜合相似度':>8}")
    print("-" * 120)

    for result in comparison_results:
        link_display = result['link'][:60] + '...' if len(result['link']) > 63 else result['link']
        # 如果文本無意義，添加標註
        text_sim_display = f"{result['text_similarity']:>14.3f}"
        if 'text_meaningful' in result and not result['text_meaningful']:
            text_sim_display = f"{result['text_similarity']:>14.3f}*"  # 添加星號標註
        
        print(f"{link_display:<62} "
              f"{result['audio_similarity']:>14.3f} "
              f"{result['image_similarity']:>14.3f} "
              f"{text_sim_display} "
              f"{result['overall_similarity']:>14.3f}")
    print("-" * 120)
    # 添加註解說明
    if any(not r.get('text_meaningful', True) for r in comparison_results):
        print("\n註: * 表示該影片的文本內容被判定為無意義，其文本相似度權重已被重新分配到音訊和畫面相似度")