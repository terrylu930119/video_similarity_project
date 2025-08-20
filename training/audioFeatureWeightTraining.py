import csv
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
import optuna
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


from utils.downloader import download_video, generate_safe_filename
from utils.logger import logger
from core.audio_processor import chunkwise_dtw_sim, compute_audio_features, extract_audio, cos_sim, dtw_sim, chamfer_sim


@dataclass
class VideoRecord:
    """影片記錄"""
    url: str
    group_label: str
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    features: Optional[Dict] = None
    download_success: bool = False
    feature_extraction_success: bool = False


@dataclass
class SimilarityPair:
    """相似度對比記錄"""
    video1: VideoRecord
    video2: VideoRecord
    expected_similarity: float  # 基於標籤計算的期望相似度
    predicted_similarity: Optional[float] = None
    group_similarity: str = ""  # "same_group", "different_group"


class VideoDatasetProcessor:
    """影片數據集處理器"""

    def __init__(self, csv_file: str, output_base_dir: str = "./optimization_data"):
        self.csv_file = csv_file
        self.output_base_dir = Path(output_base_dir)
        self.videos_dir = self.output_base_dir / "videos"
        self.audio_dir = self.output_base_dir / "audio"
        self.features_dir = self.output_base_dir / "features"
        self.results_dir = self.output_base_dir / "results"

        # 創建目錄
        for dir_path in [self.videos_dir, self.audio_dir, self.features_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.video_records: List[VideoRecord] = []
        self.similarity_pairs: List[SimilarityPair] = []

    def load_csv(self) -> None:
        """載入 CSV 文件"""
        logger.info(f"載入 CSV 文件: {self.csv_file}")

        # CSV 格式：url,group_label,additional_info(可選)
        with open(self.csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row['url'].strip()
                group_label = row['group_label'].strip()

                if url and group_label:
                    self.video_records.append(VideoRecord(
                        url=url,
                        group_label=group_label
                    ))

        logger.info(f"載入了 {len(self.video_records)} 個影片記錄")

        # 統計各組數量
        group_counts = {}
        for record in self.video_records:
            group_counts[record.group_label] = group_counts.get(record.group_label, 0) + 1

        logger.info(f"群組分布: {group_counts}")

    def _check_existing_video(self, record: VideoRecord) -> bool:
        """檢查影片是否已存在"""
        safe_filename = generate_safe_filename(record.url)
        video_path = self.videos_dir / f"{safe_filename}.mp4"

        if video_path.exists() and video_path.stat().st_size > 1024 * 1024:  # > 1MB
            logger.info(f"影片已存在，跳過下載: {video_path}")
            record.video_path = str(video_path)
            record.download_success = True
            return True
        return False

    def _download_single_video(self, record: VideoRecord, resolution: str) -> VideoRecord:
        """下載單個影片"""
        try:
            # 檢查是否已存在
            if self._check_existing_video(record):
                return record

            # 下載影片
            downloaded_path = download_video(record.url, str(self.videos_dir), resolution)
            record.video_path = downloaded_path
            record.download_success = True
            logger.info(f"下載成功: {record.url} -> {downloaded_path}")

        except Exception as e:
            logger.error(f"下載失敗 {record.url}: {str(e)}")
            record.download_success = False

        return record

    def download_videos(self, max_workers: int = 3, resolution: str = "720p") -> None:
        """批量下載影片"""
        logger.info("開始批量下載影片...")

        # 並行下載
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_record = {
                executor.submit(self._download_single_video, record, resolution): record
                for record in self.video_records
            }

            for future in as_completed(future_to_record):
                record = future.result()
                # 更新記錄
                original_record = future_to_record[future]
                original_record.download_success = record.download_success
                original_record.video_path = record.video_path

        success_count = sum(1 for r in self.video_records if r.download_success)
        logger.info(f"下載完成: {success_count}/{len(self.video_records)} 成功")

    def _extract_audio_and_features(self, record: VideoRecord) -> VideoRecord:
        """提取音頻和特徵"""
        try:
            # 提取音頻
            audio_path = extract_audio(record.video_path)
            record.audio_path = audio_path

            # 提取特徵
            features = compute_audio_features(audio_path)
            if features:
                # 保存特徵到文件
                safe_filename = generate_safe_filename(record.url)
                features_file = self.features_dir / f"{safe_filename}_features.json"

                # 將 numpy 數組轉換為可序列化的格式
                serializable_features = self._make_features_serializable(features)

                with open(features_file, 'w') as f:
                    json.dump(serializable_features, f)

                record.features = features
                record.feature_extraction_success = True
                logger.info(f"特徵提取成功: {record.url}")
            else:
                logger.error(f"特徵提取失敗: {record.url}")
                record.feature_extraction_success = False

        except Exception as e:
            logger.error(f"處理音頻失敗 {record.url}: {str(e)}")
            record.feature_extraction_success = False

        return record

    def extract_audio_features(self, max_workers: int = 2) -> None:
        """批量提取音頻特徵"""
        logger.info("開始批量提取音頻特徵...")

        def process_single_video(record: VideoRecord) -> VideoRecord:
            if not record.download_success or not record.video_path:
                return record
            return self._extract_audio_and_features(record)

        # 並行處理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_record = {
                executor.submit(process_single_video, record): record
                for record in self.video_records
            }

            for future in as_completed(future_to_record):
                record = future.result()
                # 更新記錄
                original_record = future_to_record[future]
                original_record.audio_path = record.audio_path
                original_record.features = record.features
                original_record.feature_extraction_success = record.feature_extraction_success

        success_count = sum(1 for r in self.video_records if r.feature_extraction_success)
        logger.info(f"特徵提取完成: {success_count}/{len(self.video_records)} 成功")

    def _make_features_serializable(self, features: Dict) -> Dict:
        """將特徵轉換為可序列化的格式"""
        serializable = {}
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            elif isinstance(value, dict):
                serializable[key] = self._make_features_serializable(value)
            else:
                serializable[key] = value
        return serializable

    def _generate_same_group_pairs(self, groups: Dict[str, List[VideoRecord]],
                                   same_group_similarity: float, max_pairs_per_group: int) -> None:
        """生成同組內的相似度對比對"""
        for group_label, videos in groups.items():
            if len(videos) < 2:
                continue

            pairs = list(combinations(videos, 2))
            # 限制每組的對比對數量
            if len(pairs) > max_pairs_per_group:
                pairs = np.random.choice(pairs, max_pairs_per_group, replace=False)

            for video1, video2 in pairs:
                self.similarity_pairs.append(SimilarityPair(
                    video1=video1,
                    video2=video2,
                    expected_similarity=same_group_similarity,
                    group_similarity="same_group"
                ))

    def _generate_cross_group_pairs(self, groups: Dict[str, List[VideoRecord]],
                                    different_group_similarity: float, max_pairs_per_group: int) -> None:
        """生成不同組間的相似度對比對"""
        group_list = list(groups.keys())
        for i in range(len(group_list)):
            for j in range(i + 1, len(group_list)):
                group1_videos = groups[group_list[i]]
                group2_videos = groups[group_list[j]]

                # 隨機選擇一些跨組對比對
                num_cross_pairs = min(
                    max_pairs_per_group // 2,
                    len(group1_videos),
                    len(group2_videos))

                for _ in range(num_cross_pairs):
                    video1 = np.random.choice(group1_videos)
                    video2 = np.random.choice(group2_videos)

                    self.similarity_pairs.append(SimilarityPair(
                        video1=video1,
                        video2=video2,
                        expected_similarity=different_group_similarity,
                        group_similarity="different_group"
                    ))

    def generate_similarity_pairs(self,
                                  same_group_similarity: float = 0.7,
                                  different_group_similarity: float = 0.2,
                                  max_pairs_per_group: int = 50) -> None:
        """生成相似度對比對"""
        logger.info("生成相似度對比對...")

        # 只處理成功提取特徵的影片
        successful_videos = [r for r in self.video_records if r.feature_extraction_success]

        # 按群組分類
        groups = {}
        for record in successful_videos:
            if record.group_label not in groups:
                groups[record.group_label] = []
            groups[record.group_label].append(record)

        # 生成同組內的對比對（高相似度）
        self._generate_same_group_pairs(groups, same_group_similarity, max_pairs_per_group)

        # 生成不同組間的對比對（低相似度）
        self._generate_cross_group_pairs(groups, different_group_similarity, max_pairs_per_group)

        logger.info(f"生成了 {len(self.similarity_pairs)} 個相似度對比對")

        # 統計
        same_group_count = sum(
            1 for p in self.similarity_pairs if p.group_similarity == "same_group")
        different_group_count = sum(
            1 for p in self.similarity_pairs if p.group_similarity == "different_group")
        logger.info(f"同組對比對: {same_group_count}, 跨組對比對: {different_group_count}")


class WeightOptimizer:
    """權重優化器"""

    def __init__(self, similarity_pairs: List[SimilarityPair]):
        self.similarity_pairs = similarity_pairs
        self.feature_names = [
            'dl_features', 'pann_features', 'openl3_features', 'openl3_chunkwise',
            'mfcc_mean', 'mfcc_std', 'mfcc_delta_mean', 'mfcc_delta_std',
            'chroma_mean', 'chroma_std', 'onset_env', 'tempo'
        ]
        self.optimization_history = []

    def compute_similarity_with_weights(
            self, pair: SimilarityPair, weights: Dict[str, float]) -> float:
        """使用自訂權重計算相似度（優化器專用，不影響主流程）"""
        if not pair.video1.audio_path or not pair.video2.audio_path:
            return 0.0

        try:
            f1 = compute_audio_features(pair.video1.audio_path)
            f2 = compute_audio_features(pair.video2.audio_path)
            if not f1 or not f2:
                return 0.0
            return self._custom_similarity_core(f1, f2, weights)
        except Exception as e:
            logger.error(f"自訂權重比對錯誤: {e}")
            return 0.0

    def _process_onset_similarity(self, f1: Dict[str, Any], f2: Dict[str, Any],
                                  weights: Dict[str, float]) -> Tuple[List[float], List[float]]:
        """處理 onset 相似度"""
        scores, score_weights = [], []

        if 'onset_env' in f1 and 'onset_env' in f2:
            s = dtw_sim(f1['onset_env'], f2['onset_env'])
            scores.append(s)
            score_weights.append(weights.get('onset_env', 1.0))

        return scores, score_weights

    def _process_statistical_features(self, f1: Dict[str, Any], f2: Dict[str, Any],
                                      weights: Dict[str, float]) -> Tuple[List[float], List[float]]:
        """處理統計特徵相似度"""
        scores, score_weights = [], []

        for k in ['mfcc', 'mfcc_delta', 'chroma']:
            for stat in ['mean', 'std']:
                if k in f1 and k in f2 and stat in f1[k] and stat in f2[k]:
                    sim = cos_sim(f1[k][stat], f2[k][stat])**2
                    scores.append(sim)
                    score_weights.append(weights.get(f"{k}_{stat}", 1.0))

        return scores, score_weights

    def _process_tempo_similarity(self, f1: Dict[str, Any], f2: Dict[str, Any],
                                  weights: Dict[str, float]) -> Tuple[List[float], List[float]]:
        """處理節奏相似度"""
        scores, score_weights = [], []

        if 'tempo' in f1 and 'tempo' in f2:
            t1, t2 = f1['tempo'], f2['tempo']
            s1 = 1 / (1 + abs(t1['mean'] - t2['mean']) / 30)
            s2 = 1 / (1 + abs(t1['std'] - t2['std']) / 15)
            s3 = 1 / (1 + abs(t1['range'] - t2['range']) / 30)
            sim = 0.5 * s1 + 0.25 * s2 + 0.25 * s3
            scores.append(sim)
            score_weights.append(weights.get('tempo', 1.0))

        return scores, score_weights

    def _process_openl3_chunkwise(self, f1: Dict[str, Any], f2: Dict[str, Any],
                                  weights: Dict[str, float]) -> Tuple[List[float], List[float]]:
        """處理 OpenL3 chunkwise 相似度"""
        scores, score_weights = [], []

        if 'openl3_features' in f1 and 'openl3_features' in f2:
            o1, o2 = f1['openl3_features'], f2['openl3_features']
            if isinstance(o1, dict) and 'chunkwise' in o1 and 'chunkwise' in o2:
                sim = chunkwise_dtw_sim(o1['chunkwise'], o2['chunkwise'])
                scores.append(sim)
                score_weights.append(weights.get('openl3_chunkwise', 1.0))

        return scores, score_weights

    def _process_deep_features(self, f1: Dict[str, Any], f2: Dict[str, Any],
                               weights: Dict[str, float]) -> Tuple[List[float], List[float]]:
        """處理深度特徵相似度"""
        scores, score_weights = [], []

        for k in ['dl_features', 'pann_features', 'openl3_features']:
            v1, v2 = f1.get(k), f2.get(k)
            if v1 is None or v2 is None:
                continue
            try:
                if k == 'openl3_features':
                    if isinstance(v1, dict) and 'merged' in v1 and 'merged' in v2:
                        sim = cos_sim(v1['merged'], v2['merged'])
                    else:
                        continue
                elif isinstance(v1, np.ndarray) and v1.ndim == 2 and v2.ndim == 2:
                    sim = chamfer_sim(v1, v2)
                elif k == 'pann_features':
                    split = 2048
                    emb1, tag1 = v1[:split], v1[split:]
                    emb2, tag2 = v2[:split], v2[split:]
                    sim = 0.6 * cos_sim(emb1, emb2) + 0.4 * cos_sim(tag1, tag2)
                else:
                    sim = cos_sim(v1.flatten(), v2.flatten())

                scores.append(sim)
                score_weights.append(weights.get(k, 1.0))
            except Exception as e:
                logger.warning(f"特徵比對錯誤 [{k}]: {e}")

        return scores, score_weights

    def _custom_similarity_core(
            self, f1: Dict[str, Any], f2: Dict[str, Any], weights: Dict[str, float]) -> float:
        """核心比對邏輯：比對 f1、f2，並使用傳入的特徵權重表"""
        all_scores, all_weights = [], []

        # 處理各種特徵相似度
        for scores, score_weights in [
            self._process_onset_similarity(f1, f2, weights),
            self._process_statistical_features(f1, f2, weights),
            self._process_tempo_similarity(f1, f2, weights),
            self._process_openl3_chunkwise(f1, f2, weights),
            self._process_deep_features(f1, f2, weights)
        ]:
            all_scores.extend(scores)
            all_weights.extend(score_weights)

        if not all_scores:
            return 0.0

        return float(np.average(all_scores, weights=all_weights))

    def evaluate_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """評估權重性能"""
        predictions = []
        expected_values = []

        for pair in self.similarity_pairs:
            pred_sim = self.compute_similarity_with_weights(pair, weights)
            predictions.append(pred_sim)
            expected_values.append(pair.expected_similarity)

        predictions = np.array(predictions)
        expected_values = np.array(expected_values)

        # 計算評估指標
        mae = mean_absolute_error(expected_values, predictions)
        mse = np.mean((predictions - expected_values) ** 2)
        rmse = np.sqrt(mse)

        # 相關係數
        correlation, _ = spearmanr(predictions, expected_values) if len(predictions) > 1 else (0, 1)
        if np.isnan(correlation):
            correlation = 0

        # 分類準確度（將相似度轉換為高/低相似度分類）
        pred_labels = (predictions > 0.5).astype(int)
        true_labels = (expected_values > 0.5).astype(int)
        accuracy = np.mean(pred_labels == true_labels)

        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'correlation': correlation,
            'accuracy': accuracy,
            'combined_score': mae - 0.5 * correlation  # 優化目標：最小化誤差，最大化相關性
        }

        self.optimization_history.append(metrics)
        return metrics

    def bayesian_optimization(self, n_trials: int = 100) -> Dict[str, float]:
        """貝葉斯優化"""
        logger.info(f"開始貝葉斯優化，試驗次數: {n_trials}")

        def objective(trial):
            weights = {}
            for feature_name in self.feature_names:
                weights[feature_name] = trial.suggest_float(f'{feature_name}', 0.1, 3.0)

            metrics = self.evaluate_weights(weights)
            return metrics['combined_score']

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"優化完成，最佳分數: {study.best_value:.4f}")
        return study.best_params

    def analyze_results(self, best_weights: Dict[str, float], save_dir: Path) -> None:
        """分析優化結果"""
        logger.info("分析優化結果...")

        # 評估最佳權重
        final_metrics = self.evaluate_weights(best_weights)

        # 保存結果
        results = {
            'best_weights': best_weights,
            'final_metrics': final_metrics,
            'optimization_history': self.optimization_history
        }

        results_file = save_dir / 'optimization_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        # 繪製優化歷史
        self._plot_optimization_history(save_dir)

        # 生成報告
        self._generate_report(best_weights, final_metrics, save_dir)

        logger.info(f"結果已保存到: {save_dir}")

    def _plot_optimization_history(self, save_dir: Path) -> None:
        """繪製優化歷史"""
        if not self.optimization_history:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        metrics = ['mae', 'correlation', 'accuracy', 'combined_score']
        titles = [
            'Mean Absolute Error',
            'Spearman Correlation',
            'Classification Accuracy',
            'Combined Score']

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i // 2, i % 2]
            values = [h[metric] for h in self.optimization_history]
            ax.plot(values, marker='o', markersize=3)
            ax.set_title(title)
            ax.set_xlabel('Trial')
            ax.set_ylabel(metric.upper())
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / 'optimization_history.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_report(self, best_weights: Dict[str, float],
                         final_metrics: Dict[str, float],
                         save_dir: Path) -> None:
        """生成優化報告"""
        report = f"""
# 音頻特徵權重優化報告

## 最佳權重配置
```python
SIMILARITY_WEIGHTS = {{
"""
        for feature, weight in sorted(best_weights.items()):
            report += f"    '{feature}': {weight:.3f},\n"

        report += f"""}}
```

## 性能指標
- **平均絕對誤差 (MAE)**: {final_metrics['mae']:.4f}
- **均方根誤差 (RMSE)**: {final_metrics['rmse']:.4f}
- **Spearman 相關係數**: {final_metrics['correlation']:.4f}
- **分類準確度**: {final_metrics['accuracy']:.4f}

## 優化過程
- **總試驗次數**: {len(self.optimization_history)}
- **測試樣本數**: {len(self.similarity_pairs)}

## 權重分析
### 高權重特徵 (> 2.0)
"""
        high_weight_features = {k: v for k, v in best_weights.items() if v > 2.0}
        for feature, weight in sorted(high_weight_features.items(),
                                      key=lambda x: x[1], reverse=True):
            report += f"- **{feature}**: {weight:.3f}\n"

        report += f"""
### 中等權重特徵 (1.0 - 2.0)
"""
        mid_weight_features = {k: v for k, v in best_weights.items() if 1.0 <= v <= 2.0}
        for feature, weight in sorted(mid_weight_features.items(),
                                      key=lambda x: x[1], reverse=True):
            report += f"- **{feature}**: {weight:.3f}\n"

        report += f"""
### 低權重特徵 (< 1.0)
"""
        low_weight_features = {k: v for k, v in best_weights.items() if v < 1.0}
        for feature, weight in sorted(low_weight_features.items(),
                                      key=lambda x: x[1], reverse=True):
            report += f"- **{feature}**: {weight:.3f}\n"

        # 保存報告
        with open(save_dir / 'optimization_report.md', 'w', encoding='utf-8') as f:
            f.write(report)


def main_pipeline(csv_file: str, output_dir: str = "./optimization_data", n_trials: int = 100):
    """主要優化流程"""
    logger.info("開始音頻權重自動優化流程")

    # 1. 初始化處理器
    processor = VideoDatasetProcessor(csv_file, output_dir)

    # 2. 載入 CSV
    processor.load_csv()

    # 3. 下載影片
    processor.download_videos(max_workers=3, resolution="720p")

    # 4. 提取音頻特徵
    processor.extract_audio_features(max_workers=2)

    # 5. 生成相似度對比對
    processor.generate_similarity_pairs()

    # 6. 權重優化
    optimizer = WeightOptimizer(processor.similarity_pairs)
    best_weights = optimizer.bayesian_optimization(n_trials=n_trials)

    # 7. 分析結果
    optimizer.analyze_results(best_weights, processor.results_dir)

    logger.info("優化流程完成！")
    logger.info(f"最佳權重: {best_weights}")

    return best_weights, processor, optimizer


# 使用範例
if __name__ == "__main__":
    # CSV 格式範例:
    # url,group_label
    # https://www.youtube.com/watch?v=xxx,pop_music
    # https://www.youtube.com/watch?v=yyy,pop_music
    # https://www.youtube.com/watch?v=zzz,classical

    csv_file = "training/video_dataset.csv"
    best_weights, processor, optimizer = main_pipeline(csv_file, n_trials=50)

    print("優化完成！檢查 ./optimization_data/results/ 目錄查看詳細結果。")
