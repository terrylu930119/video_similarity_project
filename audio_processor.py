import ffmpeg
from logger import logger
import os
import time
import numpy as np
import librosa
import openl3
from typing import Dict, Optional
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from librosa.sequence import dtw

# 全局緩存
_feature_cache: Dict[str, np.ndarray] = {}

@lru_cache(maxsize=32)
def load_audio(audio_path: str) -> Optional[tuple]:
    try:
        y, sr = librosa.load(audio_path, sr=None)
        return y, sr
    except Exception as e:
        logger.error(f"載入音頻文件失敗 {audio_path}: {str(e)}")
        return None

def extract_openl3_embedding(audio_path: str, embedding_size: int = 512) -> Optional[np.ndarray]:
    try:
        y, sr = load_audio(audio_path)
        if y is None:
            return None
        emb, ts = openl3.get_audio_embedding(y, sr, embedding_size=embedding_size)
        return emb
    except Exception as e:
        logger.error(f"使用 OpenL3 提取音頻嵌入失敗 {audio_path}: {str(e)}")
        return None

def parallel_feature_extraction(audio_data: np.ndarray, sr: int) -> dict:
    try:
        segment_length = sr * 10
        segments = [audio_data[i:i + segment_length] for i in range(0, len(audio_data), segment_length)]

        def extract_segment_features(segment):
            mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
            chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
            return {
                'mel': np.mean(mel_spec_db, axis=1),
                'mfcc': np.mean(mfcc, axis=1),
                'chroma': np.mean(chroma, axis=1)
            }

        with ThreadPoolExecutor(max_workers=4) as executor:
            segment_features = list(executor.map(extract_segment_features, segments))

        combined_features = {
            'mel': np.mean([f['mel'] for f in segment_features], axis=0),
            'mfcc': np.mean([f['mfcc'] for f in segment_features], axis=0),
            'chroma': np.mean([f['chroma'] for f in segment_features], axis=0)
        }

        return combined_features
    except Exception as e:
        logger.error(f"特徵提取失敗: {str(e)}")
        return None

def apply_pca(embeddings: np.ndarray, n_components: int = 50):
    # 如果輸入維度小於目標維度，直接返回原始數據
    if embeddings.shape[0] < n_components or embeddings.shape[1] < n_components:
        return embeddings
    pca = PCA(n_components=min(n_components, embeddings.shape[0], embeddings.shape[1]))
    reduced_embeddings = pca.fit_transform(embeddings)
    return reduced_embeddings

def apply_tsne(embeddings: np.ndarray, n_components: int = 2):
    tsne = TSNE(n_components=n_components)
    reduced_embeddings = tsne.fit_transform(embeddings)
    return reduced_embeddings

def compute_audio_features(audio_path: str) -> Optional[np.ndarray]:
    global _feature_cache
    try:
        if audio_path in _feature_cache:
            return _feature_cache[audio_path]

        audio_data = load_audio(audio_path)
        if audio_data is None:
            return None

        y, sr = audio_data
        features = parallel_feature_extraction(y, sr)
        if features is None:
            return None

        # 合併特徵並進行標準化
        combined_features = np.concatenate([
            features['mel'],
            features['mfcc'],
            features['chroma']
        ])

        # 標準化特徵
        combined_features = (combined_features - np.mean(combined_features)) / (np.std(combined_features) + 1e-8)
        
        # 存儲為 1D 數組
        _feature_cache[audio_path] = combined_features

        return combined_features
    except Exception as e:
        logger.error(f"提取音頻特徵時出錯: {str(e)}")
        return None

def compute_cosine_similarity(embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
    similarity = cosine_similarity([embeddings1], [embeddings2])
    return similarity[0][0]

def compute_euclidean_distance(embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
    distance = euclidean(embeddings1, embeddings2)
    return distance

def compute_dtw_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    D, wp = dtw(embedding1, embedding2)
    return D[-1, -1]

def audio_similarity(path1: str, path2: str) -> float:
    try:
        path1 = os.path.abspath(os.path.normpath(path1))
        path2 = os.path.abspath(os.path.normpath(path2))

        logger.info(f"檢查音頻文件1: {path1}")
        if not os.path.exists(path1):
            logger.error(f"音頻文件1不存在: {path1}")
            return 0.0
        logger.info(f"檢查音頻文件2: {path2}")
        if not os.path.exists(path2):
            logger.error(f"音頻文件2不存在: {path2}")
            return 0.0

        size1 = os.path.getsize(path1)
        if size1 == 0:
            logger.error(f"音頻文件1大小為0: {path1}")
            return 0.0
        size2 = os.path.getsize(path2)
        if size2 == 0:
            logger.error(f"音頻文件2大小為0: {path2}")
            return 0.0

        logger.info(f"音頻文件1大小: {size1} bytes")
        logger.info(f"音頻文件2大小: {size2} bytes")

        with ThreadPoolExecutor(max_workers=2) as executor:
            features = list(executor.map(compute_audio_features, [path1, path2]))

        features1, features2 = features
        if features1 is None or features2 is None:
            logger.error("無法提取音頻特徵")
            return 0.0

        # 可以選擇 Cosine Similarity 或 Euclidean Distance
        similarity = compute_cosine_similarity(features1, features2)
        logger.info(f"音頻相似度: {similarity:.3f}")

        # 若需要使用 DTW 進行時間序列對齊
        dtw_distance = compute_dtw_distance(features1, features2)
        logger.info(f"DTW 距離: {dtw_distance:.3f}")

        return float(similarity)
    except Exception as e:
        logger.error(f"計算音頻相似度時出錯: {str(e)}")
        return 0.0

def extract_audio(video_path: str) -> str:
    try:
        if not os.path.exists(video_path):
            logger.error(f"影片檔案不存在: {video_path}")
            raise FileNotFoundError(f"影片檔案不存在: {video_path}")

        video_dir = os.path.dirname(os.path.abspath(video_path))
        audio_dir = os.path.join(video_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(audio_dir, f"{video_name}.wav")

        if not os.access(audio_dir, os.W_OK):
            logger.error(f"沒有輸出目錄的寫入權限: {audio_dir}")
            raise PermissionError(f"沒有輸出目錄的寫入權限: {audio_dir}")

        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            logger.info(f"音訊檔案已存在: {audio_path}")
            return audio_path

        logger.info(f"開始提取音訊: {video_path} -> {audio_path}")
        ffmpeg.input(video_path).output(
            audio_path,
            acodec='pcm_s16le',
            ac=1,
            ar=16000,
            format='wav',
            audio_bitrate='192k'
        ).overwrite_output().run(quiet=True, capture_stdout=True, capture_stderr=True)

        for _ in range(5):
            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                logger.info(f"音訊檔案已生成: {audio_path}")
                return audio_path
            time.sleep(1)

        logger.error("音訊檔案生成失敗或檔案大小為0")
        raise RuntimeError("音訊檔案生成失敗")
    except Exception as e:
        logger.error(f"音訊提取失敗: {str(e)}")
        raise
