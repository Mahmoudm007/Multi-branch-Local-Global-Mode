from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np

try:
    from sklearn.preprocessing import Normalizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
except Exception:  # pragma: no cover - reported by availability checks
    Normalizer = None
    TruncatedSVD = None
    TfidfVectorizer = None
    make_pipeline = None

from class_descriptions import description_for_class, normalize_class_name
from .progress_tracker import ensure_dir


@dataclass(frozen=True)
class TfidfSettings:
    max_features: int = 2048
    ngram_min: int = 1
    ngram_max: int = 2
    min_df: int = 1
    stop_words: str | None = "english"
    use_svd: bool = False
    svd_components: int = 256
    svd_normalize: bool = True
    random_state: int = 42


class TfidfDescriptionEncoder:
    def __init__(self, settings: TfidfSettings) -> None:
        if TfidfVectorizer is None:
            raise RuntimeError("scikit-learn is required for TF-IDF auxiliary text features")
        self.settings = settings
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            analyzer="word",
            ngram_range=(settings.ngram_min, settings.ngram_max),
            max_features=settings.max_features,
            min_df=settings.min_df,
            stop_words=settings.stop_words,
            dtype=np.float32,
        )
        self.reducer = None
        self.feature_dim = 0

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        sparse = self.vectorizer.fit_transform(texts)
        if self.settings.use_svd and TruncatedSVD is not None:
            max_rank = min(sparse.shape[0] - 1, sparse.shape[1] - 1)
            if max_rank >= 2:
                n_components = min(self.settings.svd_components, max_rank)
                if self.settings.svd_normalize and Normalizer is not None and make_pipeline is not None:
                    self.reducer = make_pipeline(
                        TruncatedSVD(n_components=n_components, random_state=self.settings.random_state),
                        Normalizer(copy=False),
                    )
                else:
                    self.reducer = TruncatedSVD(n_components=n_components, random_state=self.settings.random_state)
                dense = self.reducer.fit_transform(sparse).astype(np.float32)
            else:
                dense = sparse.toarray().astype(np.float32)
        else:
            dense = sparse.toarray().astype(np.float32)
        self.feature_dim = int(dense.shape[1])
        return dense

    def transform(self, texts: list[str]) -> np.ndarray:
        sparse = self.vectorizer.transform(texts)
        if self.reducer is not None:
            return self.reducer.transform(sparse).astype(np.float32)
        return sparse.toarray().astype(np.float32)

    def save(self, path: Path) -> None:
        ensure_dir(path.parent)
        with path.open("wb") as handle:
            pickle.dump(self, handle)


def settings_from_config(config: dict) -> TfidfSettings:
    tfidf = config.get("tfidf", {}) if isinstance(config, dict) else {}
    stop_words = tfidf.get("stop_words", "english")
    if str(stop_words).lower() in {"none", "null", ""}:
        stop_words = None
    return TfidfSettings(
        max_features=int(tfidf.get("max_features", 2048)),
        ngram_min=int(tfidf.get("ngram_min", 1)),
        ngram_max=int(tfidf.get("ngram_max", 2)),
        min_df=int(tfidf.get("min_df", 1)),
        stop_words=stop_words,
        use_svd=bool(tfidf.get("use_svd", False)),
        svd_components=int(tfidf.get("svd_components", 256)),
        svd_normalize=bool(tfidf.get("svd_normalize", True)),
        random_state=int(tfidf.get("random_state", 42)),
    )


def build_class_description_matrix(
    index_to_class: dict[int, str],
    settings: TfidfSettings,
    metadata_dir: Path | None = None,
) -> tuple[np.ndarray, TfidfDescriptionEncoder, list[dict[str, object]]]:
    class_names = [index_to_class[idx] for idx in sorted(index_to_class)]
    descriptions = [description_for_class(name) for name in class_names]
    missing = [name for name, desc in zip(class_names, descriptions) if not desc]
    if missing:
        raise ValueError(f"Missing class descriptions for: {', '.join(missing)}")
    encoder = TfidfDescriptionEncoder(settings)
    matrix = encoder.fit_transform(descriptions)
    rows: list[dict[str, object]] = []
    for idx, (name, desc, vector) in enumerate(zip(class_names, descriptions, matrix)):
        rows.append(
            {
                "class_index": idx,
                "class_name": name,
                "normalized_class_name": normalize_class_name(name),
                "description_length": len(desc),
                "embedding_dim": int(vector.shape[0]),
                "description": desc,
            }
        )
    if metadata_dir is not None:
        ensure_dir(metadata_dir)
        encoder.save(metadata_dir / "tfidf_encoder.pkl")
        np.savez_compressed(
            metadata_dir / "class_description_embedding_cache.npz",
            embeddings=matrix,
            class_names=np.array(class_names, dtype=object),
        )
        import csv

        with (metadata_dir / "class_description_embedding_cache.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return matrix.astype(np.float32), encoder, rows
