from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.processing.labs import (
    PreprocessResult,
    do_lab01,
    do_lab02,
    do_lab03,
    do_lab04,
    do_lab05,
    do_lab06,
    do_preprocess,
)

router = APIRouter(tags=["labs"])


@router.post("/preprocess", response_model=PreprocessResult)
async def preprocess(image: UploadFile = File(...)):
    if not image.filename:
        raise HTTPException(status_code=400, detail="Image filename missing")
    return await do_preprocess(image)


@router.post(
    "/labs/{lab_id}/process",
    response_model=Dict[str, Any],
)
async def process_lab(
    lab_id: int,
    image_b64: str = Form(...),
    image_mode: Literal["grayscale", "rgb"] = Form(...),
    params_json: str = Form(...),
):
    # FastAPI form fields are strings; params_json is a JSON-encoded object.
    import json

    try:
        params = json.loads(params_json) if params_json else {}
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid params_json: {e}")

    if lab_id == 1:
        return do_lab01(image_b64=image_b64, image_mode=image_mode, params=params)  # type: ignore[return-value]
    if lab_id == 2:
        return do_lab02(image_b64=image_b64, image_mode=image_mode, params=params)  # type: ignore[return-value]
    if lab_id == 3:
        return do_lab03(image_b64=image_b64, image_mode=image_mode, params=params)  # type: ignore[return-value]
    if lab_id == 4:
        return do_lab04(image_b64=image_b64, image_mode=image_mode, params=params)  # type: ignore[return-value]
    if lab_id == 5:
        return do_lab05(image_b64=image_b64, image_mode=image_mode, params=params)  # type: ignore[return-value]
    if lab_id == 6:
        return do_lab06(image_b64=image_b64, image_mode=image_mode, params=params)  # type: ignore[return-value]

    raise HTTPException(status_code=404, detail=f"Unknown lab_id: {lab_id}")


@router.get("/health")
async def health():
    return {"ok": True}

