"""
Developer tools endpoints for reranker inspection and training.

Provides endpoints for:
- /dev/search    - Search with configurable feature weights (linear or ML reranking)
- /dev/generate  - Generate synthetic training data with configurable click simulation
- /dev/train     - Retrain the LTR model from logged click data
- /dev/model     - Inspect current model (feature importances, availability)
"""

import logging
import re
import threading
import uuid as _uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from app.ml.ranking_model import RERANK_FEATURES
from app.services.repositories.training_repository import DATASETS_DIR, training_repository

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/dev", tags=["dev"])

_PROJECT_ROOT = Path(__file__).parent.parent.parent

# ---------------------------------------------------------------------------
# Background job tracking for data generation
# ---------------------------------------------------------------------------
_generate_jobs: Dict[str, Dict] = {}  # job_id -> status dict

# Track which pre-trained model is currently active (None = default reranker.json)
_active_preset_id: Optional[int] = None

_DEFAULT_WEIGHTS: Dict[str, float] = {name: 1.0 for name in RERANK_FEATURES}


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class FeatureValues(BaseModel):
    """The 7 extracted feature values for a single result."""
    semantic_score: float
    bm25_score: float
    smoothed_ctr: float
    role_match: float
    query_length: float
    title_query_overlap: float
    content_freshness: float


class DevResult(BaseModel):
    """Search result enriched with per-feature values and scoring breakdown."""
    id: str
    title: str
    short_title: Optional[str] = None
    display_title: str
    info_type: str
    path: Optional[str] = None

    # Ranking positions
    rrf_position: int    # Position from RRF fusion (before any reranking)
    final_position: int  # Position in this response (after chosen scoring)

    # Raw scores
    rrf_score: float
    linear_score: Optional[float] = None        # Set when rerank_mode includes linear
    ml_score: Optional[float] = None            # Set when rerank_mode includes ml

    # Feature breakdown
    features: FeatureValues
    shap_contributions: Optional[Dict[str, float]] = None  # SHAP values from ML model


class DevSearchResponse(BaseModel):
    query: str
    total: int
    rerank_mode: str
    feature_weights: Optional[Dict[str, float]]  # None if not using linear scoring
    model_available: bool
    results: List[DevResult]


class GenerateRequest(BaseModel):
    """
    Parameters for synthetic training data generation.

    Supply either **preset_id** (loads all settings from a saved preset) or
    **dataset_id** (pick a dataset, then tune click weights manually).
    Manual parameters always override preset values when both are supplied.

    Click simulation rules:
    - *Target page* (URL-matched): probability scales from target_click_min
      (least popular) to target_click_max (most popular)
    - *temaside / retningslinje / other*: weight × (1 / position)
    """
    # Shortcut: load everything from a saved preset
    preset_id: Optional[int] = Field(None, description="Load all settings from a saved preset")

    # Dataset selection (required if no preset_id)
    dataset_id: Optional[int] = Field(None, description="ID of the training dataset to use")

    # Generation parameters (override preset values when provided)
    top_n: int = Field(200, ge=1, description="Use top N most-viewed pages from the CSV")
    k: int = Field(15, ge=5, le=50, description="Search results logged per query")
    clear: bool = Field(False, description="Truncate existing search/click logs before generating")

    # Click simulation parameters
    target_click_min: float = Field(0.5, ge=0.0, le=1.0)
    target_click_max: float = Field(0.9, ge=0.0, le=1.0)
    temaside_click_weight: float = Field(0.35, ge=0.0, le=1.0)
    retningslinje_click_weight: float = Field(0.30, ge=0.0, le=1.0)
    other_click_weight: float = Field(0.15, ge=0.0, le=1.0)


class GenerateResponse(BaseModel):
    success: bool
    job_id: Optional[str] = None  # For polling progress
    searches_created: int = 0
    results_shown: int = 0
    clicks_created: int = 0
    skipped: int = 0
    training_groups_available: int = 0  # Groups with ≥5 results AND at least one click
    error: Optional[str] = None


class GenerateStatusResponse(BaseModel):
    """Progress status for a running generation job."""
    job_id: str
    status: str  # "running", "completed", "failed"
    current: int = 0  # Current page being processed
    total: int = 0    # Total pages to process
    progress: float = 0.0  # 0.0 - 1.0
    searches_created: int = 0
    results_shown: int = 0
    clicks_created: int = 0
    skipped: int = 0
    training_groups_available: int = 0
    error: Optional[str] = None


class TrainRequest(BaseModel):
    days_back: int = 180
    min_group_size: int = 5
    require_any_click: bool = True
    save: bool = True  # Persist trained model to disk and reload


class TrainResponse(BaseModel):
    success: bool
    groups_trained: int = 0
    rows_trained: int = 0
    error: Optional[str] = None


class ModelInfoResponse(BaseModel):
    available: bool
    active_preset_id: Optional[int] = None
    feature_names: List[str]
    feature_importances: Optional[Dict[str, float]] = None


class PretrainedModelInfo(BaseModel):
    preset_id: int
    name: str
    description: str
    active: bool


class SelectModelRequest(BaseModel):
    preset_id: int = Field(..., description="Preset ID of the pre-trained model to activate")


# ---------------------------------------------------------------------------
# Dataset / Preset models
# ---------------------------------------------------------------------------

class DatasetInfo(BaseModel):
    id: int
    name: str
    description: str
    filename: str
    row_count: int
    created_at: datetime


class UploadDatasetResponse(BaseModel):
    success: bool
    dataset: Optional[DatasetInfo] = None
    error: Optional[str] = None


class PresetInfo(BaseModel):
    id: int
    name: str
    description: str
    dataset_id: int
    dataset_name: str
    dataset_filename: str
    top_n: int
    k: int
    target_click_min: float
    target_click_max: float
    temaside_click_weight: float
    retningslinje_click_weight: float
    other_click_weight: float
    created_at: datetime


class CreatePresetRequest(BaseModel):
    name: str = Field(..., description="Unique preset name")
    description: str = Field("", description="What this preset represents")
    dataset_id: int = Field(..., description="ID of the training dataset to use")
    top_n: int = Field(200, ge=1, description="Top N pages from the CSV")
    k: int = Field(15, ge=5, le=50, description="Results per search")
    target_click_min: float = Field(0.5, ge=0.0, le=1.0)
    target_click_max: float = Field(0.9, ge=0.0, le=1.0)
    temaside_click_weight: float = Field(0.35, ge=0.0, le=1.0)
    retningslinje_click_weight: float = Field(0.30, ge=0.0, le=1.0)
    other_click_weight: float = Field(0.15, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_feature_weights(
    w_semantic: Optional[float],
    w_bm25: Optional[float],
    w_ctr: Optional[float],
    w_role: Optional[float],
    w_query_length: Optional[float],
    w_title_overlap: Optional[float],
    w_freshness: Optional[float],
) -> Optional[Dict[str, float]]:
    """
    Build a feature-weights dict from individual query params.

    Returns None if no weights were specified (caller uses defaults).
    """
    raw = {
        "semantic_score": w_semantic,
        "bm25_score": w_bm25,
        "smoothed_ctr": w_ctr,
        "role_match": w_role,
        "query_length": w_query_length,
        "title_query_overlap": w_title_overlap,
        "content_freshness": w_freshness,
    }
    if all(v is None for v in raw.values()):
        return None
    return {k: (v if v is not None else 1.0) for k, v in raw.items()}


def _linear_score(features: Dict[str, float], weights: Dict[str, float]) -> float:
    """Weighted linear combination of feature values."""
    return sum(features.get(name, 0.0) * weights.get(name, 1.0) for name in RERANK_FEATURES)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/search", response_model=DevSearchResponse)
async def dev_search(
    query: str,
    role: Optional[str] = None,
    limit: int = Query(20, ge=1, le=100),
    rerank_mode: str = Query(
        "none",
        description=(
            "Scoring mode: "
            "'none' = RRF order (no reranking), "
            "'linear' = weighted feature sum with your weights, "
            "'ml' = XGBoost model, "
            "'both' = compute both scores, rank by linear"
        ),
    ),
    # Per-feature weights (only used when rerank_mode is 'linear' or 'both')
    w_semantic: Optional[float] = Query(None, description="Weight for semantic_score (default 1.0)"),
    w_bm25: Optional[float] = Query(None, description="Weight for bm25_score (default 1.0)"),
    w_ctr: Optional[float] = Query(None, description="Weight for smoothed_ctr (default 1.0)"),
    w_role: Optional[float] = Query(None, description="Weight for role_match (default 1.0)"),
    w_query_length: Optional[float] = Query(None, description="Weight for query_length (default 1.0)"),
    w_title_overlap: Optional[float] = Query(None, description="Weight for title_query_overlap (default 1.0)"),
    w_freshness: Optional[float] = Query(None, description="Weight for content_freshness (default 1.0)"),
):
    """
    Dev search with full feature breakdown and configurable reranking.

    Supports three reranking modes:
    - **none**: results in RRF order (baseline retrieval quality)
    - **linear**: weighted sum of the 7 LTR features (use the w_* params to tune)
    - **ml**: XGBoost LambdaMART model
    - **both**: compute both; rank by linear so you can compare ml_score vs linear_score

    All results include the raw feature values and both scores when applicable.
    """
    valid_modes = {"none", "linear", "ml", "both"}
    if rerank_mode not in valid_modes:
        raise HTTPException(status_code=400, detail=f"rerank_mode must be one of {valid_modes}")

    feature_weights = _build_feature_weights(
        w_semantic, w_bm25, w_ctr, w_role, w_query_length, w_title_overlap, w_freshness
    )
    effective_weights = feature_weights or _DEFAULT_WEIGHTS

    try:
        from app.services.search.hybrid_search import hybrid_search
        from app.services.search.ml_service import ml_service

        query_lower = query.lower()
        query_keywords = set(re.findall(r"\w+", query_lower))

        # Step 1: Run BM25 + semantic + RRF fusion to get ranked candidates
        candidates = hybrid_search._score_with_rrf(
            query, role, query_lower, query_keywords, k=limit * 3
        )
        if not candidates:
            return DevSearchResponse(
                query=query,
                total=0,
                rerank_mode=rerank_mode,
                feature_weights=feature_weights,
                model_available=ml_service.is_ranking_available(),
                results=[],
            )

        # Record RRF positions before any reranking
        for idx, c in enumerate(candidates):
            c.pre_rerank_position = idx + 1

        # Step 2: Extract the 7 LTR features for every candidate
        ctr_map = ml_service.get_ctr_map()
        features_list = [
            hybrid_search._extract_ranking_features(c, role, ctr_map, query_lower, query_keywords)
            for c in candidates
        ]

        # Step 3: Compute scores according to rerank_mode
        model_available = ml_service.is_ranking_available()
        ml_scores: List[Optional[float]] = [None] * len(candidates)
        shap_list: List[Optional[Dict[str, float]]] = [None] * len(candidates)
        linear_scores: List[Optional[float]] = [None] * len(candidates)

        if rerank_mode in ("ml", "both") and model_available:
            results_with_contribs = ml_service.get_ranking_scores_with_contributions(features_list)
            for i, (score, contribs) in enumerate(results_with_contribs):
                ml_scores[i] = score
                shap_list[i] = contribs
        elif rerank_mode in ("ml", "both") and not model_available:
            logger.warning("ML model not available for dev/search; falling back to RRF order")

        if rerank_mode in ("linear", "both"):
            for i, feat in enumerate(features_list):
                linear_scores[i] = _linear_score(feat, effective_weights)

        # Step 4: Sort by the primary score for this mode
        if rerank_mode == "ml" and model_available:
            order = sorted(range(len(candidates)), key=lambda i: -(ml_scores[i] or 0.0))
        elif rerank_mode in ("linear", "both"):
            order = sorted(range(len(candidates)), key=lambda i: -(linear_scores[i] or 0.0))
        else:
            order = list(range(len(candidates)))

        # Step 5: Build response (trim to limit)
        results: List[DevResult] = []
        for final_pos, i in enumerate(order[:limit], start=1):
            c = candidates[i]
            feat = features_list[i]
            results.append(
                DevResult(
                    id=c.item.id,
                    title=c.item.title,
                    short_title=c.item.short_title,
                    display_title=c.item.display_title,
                    info_type=c.item.content_type,
                    path=c.item.path,
                    rrf_position=c.pre_rerank_position,
                    final_position=final_pos,
                    rrf_score=round(c.rrf_raw, 6),
                    linear_score=round(linear_scores[i], 4) if linear_scores[i] is not None else None,
                    ml_score=round(ml_scores[i], 4) if ml_scores[i] is not None else None,
                    features=FeatureValues(**feat),
                    shap_contributions=shap_list[i],
                )
            )

        return DevSearchResponse(
            query=query,
            total=len(candidates),
            rerank_mode=rerank_mode,
            feature_weights=feature_weights,
            model_available=model_available,
            results=results,
        )

    except Exception:
        logger.exception("Dev search failed for query=%r", query)
        raise HTTPException(status_code=500, detail="Dev search failed")


def _run_generate_job(job_id: str, effective: dict):
    """Background worker for training data generation."""
    job = _generate_jobs[job_id]
    try:
        import sys
        sys.path.insert(0, str(_PROJECT_ROOT))
        from scripts.setup.generate_training_data import (
            parse_page_view_csv,
            match_csv_to_content,
            generate_training_data as _run_generation,
            get_connection,
        )

        dataset = training_repository.get_dataset(effective["dataset_id"])
        csv_path = DATASETS_DIR / dataset["filename"]
        csv_rows = parse_page_view_csv(str(csv_path))
        matched = match_csv_to_content(csv_rows)
        matched.sort(key=lambda x: -x["views"])
        pages = matched[: effective["top_n"]]

        job["total"] = len(pages)

        def on_progress(p):
            job["current"] = p["current"]
            job["total"] = p["total"]
            job["searches_created"] = p["searches_created"]
            job["results_shown"] = p["results_shown"]
            job["clicks_created"] = p["clicks_created"]
            job["skipped"] = p["skipped"]

        conn = get_connection()
        try:
            if effective["clear"]:
                cursor = conn.cursor()
                cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
                for tbl in ("click_logs", "search_results_shown", "search_logs"):
                    cursor.execute(f"TRUNCATE TABLE {tbl}")
                cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
                conn.commit()
                cursor.close()

            stats = _run_generation(
                conn, pages,
                k=effective["k"],
                target_click_min=effective["target_click_min"],
                target_click_max=effective["target_click_max"],
                temaside_click_weight=effective["temaside_click_weight"],
                retningslinje_click_weight=effective["retningslinje_click_weight"],
                other_click_weight=effective["other_click_weight"],
                progress_callback=on_progress,
            )

            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT sl.search_id
                    FROM search_logs sl
                    INNER JOIN click_logs cl ON sl.search_id = cl.search_id
                    INNER JOIN search_results_shown srs ON sl.search_id = srs.search_id
                    GROUP BY sl.search_id
                    HAVING COUNT(DISTINCT srs.id) >= 5
                ) AS t
                """
            )
            training_groups = cursor.fetchone()[0]
            cursor.close()
        finally:
            conn.close()

        job.update({
            "status": "completed",
            "current": job["total"],
            "searches_created": stats["searches_created"],
            "results_shown": stats["results_shown"],
            "clicks_created": stats["clicks_created"],
            "skipped": stats["skipped"],
            "training_groups_available": training_groups,
        })

    except Exception as e:
        logger.exception("Training data generation failed (job %s)", job_id)
        job["status"] = "failed"
        job["error"] = str(e)


@router.post("/generate", response_model=GenerateResponse)
async def dev_generate(request: GenerateRequest = GenerateRequest()):
    """
    Start generating synthetic training data in the background.

    Returns a **job_id** immediately. Poll **GET /dev/generate/status/{job_id}**
    for progress updates.
    """
    # Resolve settings from preset
    effective = request.model_dump()

    if request.preset_id is not None:
        preset = training_repository.get_preset(request.preset_id)
        if preset is None:
            return GenerateResponse(success=False, error=f"Preset {request.preset_id} not found")
        effective["dataset_id"] = preset["dataset_id"]
        explicitly_set = request.model_fields_set
        for field in ("top_n", "k", "target_click_min", "target_click_max",
                      "temaside_click_weight", "retningslinje_click_weight", "other_click_weight"):
            # Only use preset value if caller did not explicitly provide one
            if field not in explicitly_set:
                effective[field] = preset[field]

    if effective.get("dataset_id") is None:
        return GenerateResponse(success=False, error="Provide either preset_id or dataset_id.")

    dataset = training_repository.get_dataset(effective["dataset_id"])
    if dataset is None:
        return GenerateResponse(success=False, error=f"Dataset {effective['dataset_id']} not found")

    csv_path = DATASETS_DIR / dataset["filename"]
    if not csv_path.exists():
        return GenerateResponse(
            success=False,
            error=f"CSV file '{dataset['filename']}' not found. Re-upload via POST /dev/datasets/upload.",
        )

    # Create job and start in background
    job_id = str(_uuid.uuid4())
    _generate_jobs[job_id] = {
        "status": "running",
        "current": 0,
        "total": 0,
        "searches_created": 0,
        "results_shown": 0,
        "clicks_created": 0,
        "skipped": 0,
        "training_groups_available": 0,
        "error": None,
    }

    thread = threading.Thread(target=_run_generate_job, args=(job_id, effective), daemon=True)
    thread.start()

    return GenerateResponse(success=True, job_id=job_id)


@router.get("/generate/status/{job_id}", response_model=GenerateStatusResponse)
async def dev_generate_status(job_id: str):
    """Poll progress for a running generation job."""
    job = _generate_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    total = job["total"]
    current = job["current"]
    progress = current / total if total > 0 else 0.0

    return GenerateStatusResponse(
        job_id=job_id,
        status=job["status"],
        current=current,
        total=total,
        progress=round(progress, 3),
        searches_created=job["searches_created"],
        results_shown=job["results_shown"],
        clicks_created=job["clicks_created"],
        skipped=job["skipped"],
        training_groups_available=job["training_groups_available"],
        error=job["error"],
    )


# ---------------------------------------------------------------------------
# Dataset management
# ---------------------------------------------------------------------------

@router.get("/datasets", response_model=List[DatasetInfo])
async def list_datasets():
    """List all registered training datasets."""
    rows = training_repository.get_all_datasets()
    return [DatasetInfo(**r) for r in rows]


@router.post("/datasets/upload", response_model=UploadDatasetResponse)
async def upload_dataset(
    file: UploadFile = File(..., description="CSV file with columns: title, url, ..., views"),
    name: str = Query(..., description="Unique dataset name"),
    description: str = Query("", description="Short description of this dataset"),
):
    """
    Upload a CSV file and register it as a training dataset.

    The file is saved to **data/training_datasets/** and its metadata is stored
    in the database.  Use the returned `id` in **POST /dev/generate** or when
    creating a preset.

    Expected CSV format: same as helsedir_page_view.csv
    (columns: title, url, ..., views as the 5th column).
    """
    if not file.filename or not file.filename.endswith(".csv"):
        return UploadDatasetResponse(success=False, error="File must be a .csv")

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    # Use the original filename but sanitize it
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in file.filename)
    dest = DATASETS_DIR / safe_name

    # If a file with this name already exists, add a timestamp suffix
    if dest.exists():
        stem, suffix = safe_name.rsplit(".", 1) if "." in safe_name else (safe_name, "")
        from datetime import datetime as _dt
        safe_name = f"{stem}_{_dt.now().strftime('%Y%m%d%H%M%S')}.{suffix}"
        dest = DATASETS_DIR / safe_name

    try:
        content = await file.read()
        dest.write_bytes(content)

        # Count data rows for metadata
        try:
            import sys
            sys.path.insert(0, str(_PROJECT_ROOT))
            from scripts.setup.generate_training_data import parse_page_view_csv
            rows = parse_page_view_csv(str(dest))
            row_count = len(rows)
        except Exception:
            row_count = 0

        dataset_id = training_repository.create_dataset(
            name=name,
            filename=safe_name,
            row_count=row_count,
            description=description,
        )
        dataset = training_repository.get_dataset(dataset_id)
        return UploadDatasetResponse(success=True, dataset=DatasetInfo(**dataset))

    except RuntimeError as e:
        dest.unlink(missing_ok=True)
        return UploadDatasetResponse(success=False, error=str(e))
    except Exception as e:
        dest.unlink(missing_ok=True)
        logger.exception("Dataset upload failed")
        return UploadDatasetResponse(success=False, error=str(e))


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: int):
    """
    Delete a dataset record from the database and its CSV file from disk.
    Any presets referencing this dataset are also deleted (CASCADE).
    """
    dataset = training_repository.get_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Delete from DB first, then remove file only on success
    deleted = training_repository.delete_dataset(dataset_id)
    if not deleted:
        raise HTTPException(status_code=500, detail="Failed to delete dataset from database")

    csv_path = DATASETS_DIR / dataset["filename"]
    csv_path.unlink(missing_ok=True)
    return {"deleted": True, "id": dataset_id}


# ---------------------------------------------------------------------------
# Preset management
# ---------------------------------------------------------------------------

@router.get("/presets", response_model=List[PresetInfo])
async def list_presets():
    """List all saved training presets."""
    rows = training_repository.get_all_presets()
    return [PresetInfo(**r) for r in rows]


@router.post("/presets", response_model=PresetInfo)
async def create_preset(request: CreatePresetRequest):
    """
    Save a named training preset (dataset + click simulation parameters).

    Use the preset's `id` in **POST /dev/generate** to reproduce the same
    training configuration at any time.
    """
    if training_repository.get_dataset(request.dataset_id) is None:
        raise HTTPException(status_code=404, detail=f"Dataset {request.dataset_id} not found")

    try:
        preset_id = training_repository.create_preset(**request.model_dump())
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    preset = training_repository.get_preset(preset_id)
    return PresetInfo(**preset)


@router.delete("/presets/{preset_id}")
async def delete_preset(preset_id: int):
    """Delete a saved preset."""
    if not training_repository.delete_preset(preset_id):
        raise HTTPException(status_code=404, detail="Preset not found")
    return {"deleted": True, "id": preset_id}


@router.post("/train", response_model=TrainResponse)
async def dev_train(request: TrainRequest = TrainRequest()):
    """
    Retrain the LTR ranking model from logged click data.

    Uses search_logs + search_results_shown + click_logs to build
    LambdaMART training groups. Optionally saves the new model to disk
    and reloads it so it's used immediately by /search.

    Requires at least some click data (run generate_training_data.py first
    if no real traffic has been collected yet).
    """
    try:
        from pathlib import Path
        from app.config import settings
        from app.ml.ranking_model import HealthContentReranker
        from app.services.search.ml_service import ml_service

        reranker = HealthContentReranker()
        metrics = reranker.train_from_database(
            days_back=request.days_back,
            min_group_size=request.min_group_size,
            require_any_click=request.require_any_click,
            verbose=True,
        )

        if metrics.get("trained") == 0.0:
            return TrainResponse(
                success=False,
                error=(
                    "No usable training groups found. "
                    f"Checked {request.days_back} days back, "
                    f"min_group_size={request.min_group_size}. "
                    "Run generate_training_data.py to seed click data."
                ),
            )

        if request.save:
            model_dir = Path(settings.ml_models_dir) / "ranking"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "reranker.json"
            reranker.save(str(model_path))
            ml_service.load_ranking_model(force=True)
            logger.info("Ranking model saved to %s and reloaded", model_path)

        return TrainResponse(
            success=True,
            groups_trained=int(metrics.get("groups", 0)),
            rows_trained=int(metrics.get("rows", 0)),
        )

    except Exception as e:
        logger.exception("Model training failed")
        return TrainResponse(success=False, error=str(e))


@router.get("/models", response_model=List[PretrainedModelInfo])
async def dev_list_models():
    """
    List all pre-trained models available for selection.

    Each model corresponds to a training preset and is stored in
    models/ranking/<preset_id>.json.
    """
    global _active_preset_id
    presets = training_repository.get_all_presets()
    model_dir = _PROJECT_ROOT / "models" / "ranking"

    result = []
    for p in presets:
        model_path = model_dir / f"{p['id']}.json"
        if model_path.exists():
            result.append(PretrainedModelInfo(
                preset_id=p["id"],
                name=p["name"],
                description=p["description"],
                active=(_active_preset_id == p["id"]),
            ))
    return result


@router.post("/model/select", response_model=ModelInfoResponse)
async def dev_select_model(request: SelectModelRequest):
    """
    Switch the active ranking model to a pre-trained model.

    Loads the model from models/ranking/<preset_id>.json and makes it
    the active model used by /search when rerank=true.
    """
    global _active_preset_id
    from app.services.search.ml_service import ml_service
    from app.ml.ranking_model import HealthContentReranker

    model_path = _PROJECT_ROOT / "models" / "ranking" / f"{request.preset_id}.json"
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No pre-trained model for preset {request.preset_id}. "
                   f"Run: python scripts/setup/pretrain_all_models.py --preset {request.preset_id}",
        )

    try:
        reranker = HealthContentReranker.load(str(model_path))
        ml_service.ranking_model = reranker
        _active_preset_id = request.preset_id
        logger.info("Switched to pre-trained model for preset %d", request.preset_id)
    except Exception:
        logger.exception("Failed to load model for preset %d", request.preset_id)
        raise HTTPException(status_code=500, detail="Failed to load model")

    # Return model info
    try:
        raw = reranker.model.get_booster().get_score(importance_type="gain")
        total = sum(raw.values()) or 1.0
        importances: Dict[str, float] = {name: 0.0 for name in RERANK_FEATURES}
        for k, v in raw.items():
            if k in importances:
                importances[k] = round(v / total, 4)
    except Exception:
        importances = None

    return ModelInfoResponse(
        available=True,
        active_preset_id=_active_preset_id,
        feature_names=list(RERANK_FEATURES),
        feature_importances=importances,
    )


@router.get("/model", response_model=ModelInfoResponse)
async def dev_model_info():
    """
    Get information about the currently loaded ranking model.

    Returns feature importances (normalized gain) so the frontend can
    show which features the model relies on most.
    """
    global _active_preset_id
    from app.services.search.ml_service import ml_service

    if not ml_service.is_ranking_available():
        ml_service.load_ranking_model(force=True)

    ranking_model = ml_service.ranking_model
    if ranking_model is None or ranking_model.model is None:
        return ModelInfoResponse(available=False, feature_names=list(RERANK_FEATURES))

    try:
        raw = ranking_model.model.get_booster().get_score(importance_type="gain")
        total = sum(raw.values()) or 1.0
        importances: Dict[str, float] = {name: 0.0 for name in RERANK_FEATURES}
        for k, v in raw.items():
            if k in importances:
                importances[k] = round(v / total, 4)
    except Exception:
        logger.exception("Could not compute feature importances")
        importances = None

    return ModelInfoResponse(
        available=True,
        active_preset_id=_active_preset_id,
        feature_names=list(RERANK_FEATURES),
        feature_importances=importances,
    )
