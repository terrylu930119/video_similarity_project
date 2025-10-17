# core/similarity.py
"""
檔案用途：多模態相似度整合與結果處理

此模組提供影音比對系統的核心相似度計算功能，包括：
- 多模態相似度整合（音訊、影像、文本）
- 權重配置與加權計算
- 結果格式化與輸出
- 檔案驗證與錯誤處理

主要功能：
- calculate_similarity: 主要相似度計算函式
- SimilarityResult: 相似度結果資料結構
- 各種輔助工具函式
"""

import os
import torch
from utils.logger import logger
from core.text import text_similarity
from core.audio import audio_similarity
from core.image import video_similarity
from typing import List, Dict, Union, Optional, TypedDict, Callable, Tuple


# =============== 型別宣告 ===============
class SimilarityResult(TypedDict):
    """相似度比對結果資料結構。

    此類別定義了影音比對結果的標準格式，包含各模態的相似度分數
    以及整體綜合評分，用於 CLI 輸出與 API 回應。

    Attributes:
        link (str): 影片或資源連結（顯示用）
        audio_similarity (float): 音訊相似度分數（0~1）
        image_similarity (float): 影像相似度分數（0~1）
        text_similarity (float): 文本相似度分數（0~1）
        overall_similarity (float): 綜合加權後分數（0~1）
        text_meaningful (Optional[bool]): 文本是否有效（若 False，text 權重會被重配）
        text_status (Optional[str]): 文本判定說明（無語意/長度不足/錯誤等）

    Note:
        - 所有相似度分數範圍為 0~1，數值越高表示越相似
        - text_meaningful 為 False 時，文本權重會重新分配給其他模態
    """
    link: str
    audio_similarity: float
    image_similarity: float
    text_similarity: float
    overall_similarity: float
    text_meaningful: Optional[bool]
    text_status: Optional[str]


ProgressCB = Optional[Callable[..., None]]


# =============== 路徑處理與基礎檢查 ===============
def _abs_file(p: str) -> str:
    """將任意路徑正規化為絕對路徑。

    功能：
        - 將相對路徑轉換為絕對路徑
        - 正規化路徑格式（移除多餘的分隔符號）
        - 確保路徑一致性，避免存在性檢查與 I/O 差異

    Args:
        p (str): 輸入路徑（相對或絕對）

    Returns:
        str: 正規化後的絕對路徑

    Note:
        - 使用 os.path.abspath 和 os.path.normpath 確保路徑正確性
        - 避免相對路徑導致的檔案找不到問題
    """
    return os.path.abspath(os.path.normpath(p))


def check_files_exist(files: Union[str, List[str]], check_size: bool = True) -> bool:
    """驗證檔案是否存在且（可選）大小 > 0。

    功能：
        - 檢查檔案是否存在於檔案系統
        - 驗證檔案是否為一般檔案（非目錄）
        - 可選的檔案大小檢查（避免空檔案）
        - 預先過濾無效檔案，提升下游模型魯棒性

    Args:
        files (Union[str, List[str]]): 要檢查的檔案路徑（單一或清單）
        check_size (bool, optional): 是否檢查檔案大小。預設為 True。

    Returns:
        bool: 是否全部檔案通過檢查

    Note:
        - 下游模型對空檔或目錄會崩潰，此函式提供預先過濾
        - 會記錄錯誤訊息到日誌
        - 支援單一檔案或檔案清單輸入
    """
    # ──────────────── 第1階段：統一為 list 格式 ────────────────
    if isinstance(files, str):
        files = [files]

    # ──────────────── 第2階段：逐一檢查存在性與型態 ────────────────
    for file in files:
        try:
            abs_path = _abs_file(file)
            if not os.path.exists(abs_path):
                logger.error(f"文件不存在: {abs_path}")
                return False
            if not os.path.isfile(abs_path):
                logger.error(f"路徑不是文件: {abs_path}")
                return False

            # ──────────────── 第3階段：（選）檔案大小檢查 ────────────────
            if check_size:
                try:
                    if os.path.getsize(abs_path) == 0:
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
    """從檔名推導 video_id。

    功能：
        - 從檔案路徑中提取 video_id 前綴
        - 支援不同檔案類型的命名規則
        - 用於配對正確的幀集與資源

    Args:
        file_path (str): 檔案路徑

    Returns:
        str: 提取的 video_id

    Note:
        命名規則：
        - xxx_frame_*.jpg → 回傳 'xxx'
        - xxx_transcript.* → 回傳 'xxx'
        - 其他 → 去副檔名的 basename
    """
    basename = os.path.basename(file_path)
    if '_frame_' in basename:
        return basename.split('_frame_')[0]
    elif '_transcript' in basename:
        return basename.split('_transcript')[0]
    else:
        return os.path.splitext(basename)[0]


def _select_and_abs_image_paths(image: Union[str, List[str]], video_id: str) -> List[str]:
    """從（單檔或清單）中篩出屬於指定 video_id 的幀，並轉為絕對路徑。

    功能：
        - 篩選出屬於指定 video_id 的幀檔案
        - 將相對路徑轉換為絕對路徑
        - 避免混入其他影片的幀，確保相似度計算正確性

    Args:
        image (Union[str, List[str]]): 幀檔案路徑（單一或清單）
        video_id (str): 目標 video_id

    Returns:
        List[str]: 篩選並轉換後的絕對路徑清單

    Note:
        - 避免混入其他影片的幀，確保相似度只在對應影片之間計算
    """
    # ──────────────── 第1階段：依輸入型別處理 ────────────────
    if isinstance(image, list):
        return [_abs_file(p) for p in image if get_video_id_from_path(p) == video_id]

    # ──────────────── 第2階段：單一路徑情況 ────────────────
    return [_abs_file(image)]


# =============== 權重策略與分數融合 ===============
def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    """權重歸一化，避免權重和 != 1 造成整體分數偏差。

    功能：
        - 將權重字典歸一化，確保總和為 1
        - 處理邊界情況，避免除零錯誤
        - 保持權重比例關係

    Args:
        w (Dict[str, float]): 原始權重字典

    Returns:
        Dict[str, float]: 歸一化後的權重字典

    Note:
        - 若總和極小（例如皆為 0），使用極小常數避免 ZeroDivisionError
    """
    # ──────────────── 第1階段：計算總和並保底 ────────────────
    s = max(w.get("audio", 0.0) + w.get("image", 0.0) + w.get("text", 0.0), 1e-9)

    # ──────────────── 第2階段：回傳歸一化結果 ────────────────
    return {
        "audio": w.get("audio", 0.0) / s,
        "image": w.get("image", 0.0) / s,
        "text": w.get("text", 0.0) / s,
    }


def _redistribute_for_text_skip(w: Dict[str, float]) -> Dict[str, float]:
    """當文本被判定為無效時，將 text 權重依比例分配到 audio/image。

    功能：
        - 將 text 權重重新分配給 audio 和 image
        - 保持總權重一致性
        - 避免文本雜訊影響最終分數

    Args:
        w (Dict[str, float]): 原始權重字典

    Returns:
        Dict[str, float]: 重新分配後的權重字典

    Note:
        - 避免文本雜訊拉低最終分數，同時保留總權重一致性
        - 按比例分配，若 audio 和 image 權重皆為 0 則平分
    """
    # ──────────────── 第1階段：計算剩餘可分配比例 ────────────────
    text_w = w.get("text", 0.0)
    remain = w.get("audio", 0.0) + w.get("image", 0.0)

    # ──────────────── 第2階段：比例分配 / 平分回退 ────────────────
    if remain > 0:
        w2 = {
            "audio": w["audio"] + text_w * (w["audio"] / remain if remain else 0.5),
            "image": w["image"] + text_w * (w["image"] / remain if remain else 0.5),
            "text": 0.0
        }
    else:
        # 兩者皆 0 的極端情況，平分文本權重
        w2 = {"audio": text_w * 0.5, "image": text_w * 0.5, "text": 0.0}

    # ──────────────── 第3階段：再做一次歸一化（安全保險） ────────────────
    return _normalize_weights(w2)


def _blend(a: float, i: float, t: float, w: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """依權重將三模態分數做線性加權。

    功能：
        - 將音訊、影像、文本相似度分數進行線性加權
        - 權重自動歸一化，確保總和為 1
        - 提供統一的融合策略，便於後續替換為學習式權重

    Args:
        a (float): 音訊相似度分數（0~1）
        i (float): 影像相似度分數（0~1）
        t (float): 文本相似度分數（0~1）
        w (Dict[str, float]): 權重配置字典

    Returns:
        Tuple[float, Dict[str, float]]: (綜合相似度分數, 歸一化後的權重)

    Note:
        - 權重會自動歸一化，避免總和不為 1 的問題
        - 提供統一的融合策略，便於後續擴展
    """
    # ──────────────── 第1階段：權重歸一化 ────────────────
    wN = _normalize_weights(w)

    # ──────────────── 第2階段：線性加權 ────────────────
    overall = a * wN["audio"] + i * wN["image"] + t * wN["text"]
    return float(overall), wN


# =============== 各模態計分：音訊 ===============
def _score_audio(audio1: str, audio2: str, emit: ProgressCB, task_id: Optional[str]) -> float:
    """計算兩段音訊的相似度。

    功能：
        - 計算兩段音訊檔案的相似度分數
        - 支援進度回報與錯誤處理
        - 檔案有效性檢查

    Args:
        audio1 (str): 第一段音訊檔案路徑
        audio2 (str): 第二段音訊檔案路徑
        emit (ProgressCB): 進度回報回調函式
        task_id (Optional[str]): 任務 ID

    Returns:
        float: 音訊相似度分數（0~1）

    Note:
        處理流程：回報進度 → 檔案檢查 → 核心計算 → 錯誤處理
    """
    # ──────────────── 第1階段：回報進度 ────────────────
    if emit and task_id:
        emit("progress", task_id=task_id, phase="audio", percent=60, msg="開始計算音頻相似度")

    # ──────────────── 第2階段：檔案檢查 ────────────────
    a1, a2 = _abs_file(audio1), _abs_file(audio2)
    if not check_files_exist([a1, a2], check_size=True):
        logger.error("音頻文件無效，跳過音頻相似度計算")
        return 0.0

    # ──────────────── 第3階段：核心相似度計算 ────────────────
    try:
        logger.info(f"開始計算音頻相似度: {a1} 和 {a2}")
        val = float(audio_similarity(a1, a2))
        logger.info(f"音頻相似度: {val:.3f}")

        # 回報分數到前端，增加可觀測性
        if emit and task_id:
            emit("progress", task_id=task_id, phase="audio", percent=80, msg=f"音頻相似度: {val:.3f}")
        return val

    # ──────────────── 第4階段：錯誤處理 ────────────────
    except Exception as e:
        logger.error(f"計算音頻相似度時出錯: {str(e)}")
        return 0.0


# =============== 各模態計分：圖像 ===============
def _score_image(
    image1: Union[str, List[str]], image2: Union[str, List[str]],
    audio1: str, audio2: str, video_duration: float,
    gpu: bool, emit: ProgressCB, task_id: Optional[str]
) -> float:
    """計算兩段影片（或幀序列）的圖像相似度。

    功能：
        - 計算兩段影片的圖像相似度分數
        - 支援 GPU 加速與進度回報
        - 自動篩選對應的幀集

    Args:
        image1 (Union[str, List[str]]): 第一段影片的幀檔案（單一或清單）
        image2 (Union[str, List[str]]): 第二段影片的幀檔案（單一或清單）
        audio1 (str): 第一段音訊檔案路徑（用於推導 video_id）
        audio2 (str): 第二段音訊檔案路徑（用於推導 video_id）
        video_duration (float): 影片時長（秒）
        gpu (bool): 是否使用 GPU 加速
        emit (ProgressCB): 進度回報回調函式
        task_id (Optional[str]): 任務 ID

    Returns:
        float: 圖像相似度分數（0~1）

    Note:
        - 藉由音訊檔名推導對應的 video_id 以篩選幀集
        - 避免混入其他影片的幀
    """
    # ──────────────── 第1階段：回報進度 ────────────────
    if emit and task_id:
        emit("progress", task_id=task_id, phase="image", percent=80, msg="開始計算圖像相似度")

    # ──────────────── 第2階段：以 video_id 配對幀集 ────────────────
    v1, v2 = get_video_id_from_path(audio1), get_video_id_from_path(audio2)
    p1 = _select_and_abs_image_paths(image1, v1)
    p2 = _select_and_abs_image_paths(image2, v2)

    # ──────────────── 第3階段：幀檔檢查 ────────────────
    if not check_files_exist(p1, True) or not check_files_exist(p2, True):
        logger.error("幀文件無效，跳過圖像相似度計算")
        return 0.0

    # ──────────────── 第4階段：核心相似度計算 ────────────────
    try:
        logger.info(f"開始計算圖像相似度: {len(p1)} 和 {len(p2)} 幀")
        # GPU 時可適度放大 batch size；CPU 降低以避免 OOM
        batch_size = 64 if gpu else 32
        res = video_similarity(p1, p2, video_duration, batch_size)
        val = float(res.get("similarity", 0.0))
        logger.info(f"圖像相似度: {val:.3f}")

        if emit and task_id:
            emit("progress", task_id=task_id, phase="image", percent=90, msg=f"圖像相似度: {val:.3f}")
        return val

    # ──────────────── 第5階段：錯誤處理 ────────────────
    except Exception as e:
        logger.error(f"計算圖像相似度時出錯: {str(e)}")
        return 0.0


# =============== 各模態計分：文本 ===============
def _score_text(text1: str, text2: str, emit: ProgressCB, task_id: Optional[str]) -> Tuple[float, bool, str]:
    """計算兩段文本的相似度，並判斷文本是否具語意（有效）。

    功能：
        - 計算兩段文本的語義相似度分數
        - 判斷文本是否具有語意價值
        - 支援進度回報與錯誤處理

    Args:
        text1 (str): 第一段文本內容
        text2 (str): 第二段文本內容
        emit (ProgressCB): 進度回報回調函式
        task_id (Optional[str]): 任務 ID

    Returns:
        Tuple[float, bool, str]: (文本相似度分數, 是否有效, 原因說明)

    Note:
        - 若字幕/轉寫內容噪聲過高，將不納入加權計算，避免污染總分
    """
    # ──────────────── 第1階段：回報進度 ────────────────
    if emit and task_id:
        emit("progress", task_id=task_id, phase="text", percent=95, msg="開始計算文本相似度")

    # ──────────────── 第2階段：核心相似度計算 ────────────────
    try:
        t_sim, is_meaningful, reason = text_similarity(text1, text2)
        logger.info(f"文本相似度: {t_sim:.3f} ({reason})")

        # 回報分數到前端，便於 UI 呈現
        if emit and task_id:
            emit("progress", task_id=task_id, phase="text", percent=99, msg=f"文本相似度: {t_sim:.3f}")
        return float(t_sim), bool(is_meaningful), str(reason)

    # ──────────────── 第3階段：錯誤處理 ────────────────
    except Exception as e:
        logger.error(f"計算文本相似度時出錯: {str(e)}")
        return 0.0, False, "錯誤發生"


# =============== 主入口：整體相似度計算 ===============
def calculate_overall_similarity(
    audio1: str, audio2: str,
    image1: Union[str, List[str]], image2: Union[str, List[str]],
    text1: str, text2: str, video_duration: float,
    weights: Optional[Dict[str, float]] = None,
    emit_cb: ProgressCB = None, task_id: Optional[str] = None,
    link: Optional[str] = None
) -> Dict[str, Union[float, bool, str, Dict[str, float]]]:
    """協調三個模態（音訊/圖像/文本）的相似度計算與權重融合，產生最終綜合分數。

    功能：
        - 協調音訊、影像、文本三個模態的相似度計算
        - 處理權重配置與分數融合
        - 支援進度回報與錯誤處理
        - 提供統一的對外 API 介面

    Args:
        audio1 (str): 第一段音訊檔案路徑
        audio2 (str): 第二段音訊檔案路徑
        image1 (Union[str, List[str]]): 第一段影片的幀檔案（單一或清單）
        image2 (Union[str, List[str]]): 第二段影片的幀檔案（單一或清單）
        text1 (str): 第一段文本內容
        text2 (str): 第二段文本內容
        video_duration (float): 影片時長（秒）
        weights (Optional[Dict[str, float]], optional): 權重配置。預設為 None。
        emit_cb (ProgressCB, optional): 進度回報回調函式。預設為 None。
        task_id (Optional[str], optional): 任務 ID。預設為 None。
        link (Optional[str], optional): 資源連結。預設為 None。

    Returns:
        Dict[str, Union[float, bool, str, Dict[str, float]]]: 相似度計算結果

    Note:
        - 作為協調層，讓各模態實作可獨立替換而不影響對外 API
        - 支援 GPU 加速與進度回報
        - 包含各模態分數、文本有效性、最終權重與總分
    """

    # ──────────────── 第1階段：初始化與環境探測 ────────────────
    # 未指定權重則使用預設；並檢測是否可用 GPU，調整批量與日誌
    if weights is None:
        weights = {'audio': 0.3, 'image': 0.4, 'text': 0.3}
    gpu_available = torch.cuda.is_available()
    logger.info("使用 GPU 加速相似度計算" if gpu_available else "使用 CPU 進行相似度計算")

    # ──────────────── 第2階段：各模態相似度計算 ────────────────
    a_sim = _score_audio(audio1, audio2, emit_cb, task_id)
    i_sim = _score_image(image1, image2, audio1, audio2, video_duration, gpu_available, emit_cb, task_id)
    t_sim, is_meaningful, reason = _score_text(text1, text2, emit_cb, task_id)

    # ──────────────── 第3階段：權重調整（文本可能無效） ────────────────
    eff_weights = dict(weights)  # copy，避免修改呼叫方傳入的物件
    if not is_meaningful:
        logger.warning(f"文本相似度計算跳過: {reason}")
        eff_weights = _redistribute_for_text_skip(eff_weights)
        logger.info(
            "權重重新分配（文本無效） → 音訊=%.3f, 圖像=%.3f, 文本=%.3f",
            eff_weights['audio'], eff_weights['image'], eff_weights['text']
        )

    # ──────────────── 第4階段：融合與回傳 ────────────────
    overall, eff_weights = _blend(
        a_sim, i_sim, t_sim if is_meaningful else 0.0, eff_weights
    )

    logger.info(
        "整體相似度: %.3f (音頻=%.3f*%.2f, 圖像=%.3f*%.2f, 文本=%.3f*%.2f)",
        overall, a_sim, eff_weights['audio'], i_sim, eff_weights['image'], t_sim, eff_weights['text']
    )

    return {
        "audio_similarity": a_sim,
        "image_similarity": i_sim,
        "text_similarity": t_sim,
        "overall_similarity": overall,
        "weights": eff_weights,
        "text_meaningful": is_meaningful,
        "text_status": reason
    }


# =============== CLI 輸出（可選） ===============
def display_similarity_results(reference_link: str, comparison_results: List[SimilarityResult]) -> None:
    """將多筆比對結果以表格形式輸出（CLI 友好）。

    功能：
        - 以表格形式顯示相似度比對結果
        - 支援 CLI 友好的格式化輸出
        - 標註無效文本的相似度分數

    Args:
        reference_link (str): 參考影片連結
        comparison_results (List[SimilarityResult]): 比對結果清單

    Note:
        - 便於本地除錯或報表匯出前的人工檢視
        - 若 text_meaningful=False，會在文本相似度後加上 * 以示區別
    """
    # ──────────────── 第1階段：表頭輸出 ────────────────
    print("\n=== 相似度比對結果 ===")
    print(f"參考影片: {reference_link[:60] + '...' if len(reference_link) > 63 else reference_link}\n")
    print("比對結果:")
    print("-" * 122)
    print(f"{'影片連結':<62} {'音訊相似度':>14} {'畫面相似度':>14} {'內容相似度':>14} {'綜合相似度':>14}")
    print("-" * 122)

    # ──────────────── 第2階段：逐列輸出 ────────────────
    for result in comparison_results:
        link_display = result['link'][:60] + '...' if len(result['link']) > 63 else result['link']
        text_sim_display = f"{result['text_similarity']:>14.3f}"
        if result.get('text_meaningful') is False:
            text_sim_display = f"{result['text_similarity']:>14.3f}*"

        print(
            f"{link_display:<62} "
            f"{result['audio_similarity']:>14.3f} "
            f"{result['image_similarity']:>14.3f} "
            f"{text_sim_display} "
            f"{result['overall_similarity']:>14.3f}"
        )

    # ──────────────── 第3階段：備註說明 ────────────────
    print("-" * 122)
    if any(r.get('text_meaningful') is False for r in comparison_results):
        print("\n註: * 表示該影片的文本內容被判定為無意義，其文本權重已被重新分配到音訊與畫面")
