"""Product-inspection logic: optimize images, call OpenAI, parse the verdict.

Pure, GUI-free port of the OpenAI half of integrated_oak_inspection.py. Knows
nothing about cameras or ROS -- it takes a list of image paths plus a product
name and returns a structured inspection verdict
(good / damaged / manual_review) with a confidence.

The OpenAI API key is read from the OPENAI_API_KEY environment variable only;
it is never stored in config. `run_inspection(cfg, image_paths, product_name)`
is the single entry point used by the InspectionDetector and the CLI.
"""

import base64
import json
import os
from pathlib import Path

import requests
from PIL import Image, ImageOps


def build_prompt(product_name):
    return "\n".join([
        "You are a warehouse return-inspection assistant for consumer products.",
        "Your job is to evaluate ONLY what is visibly present in the provided images.",
        "Do not guess about hidden, internal, or non-visible damage.",
        "Do not assume functionality unless visible evidence strongly supports a conclusion.",
        "",
        "Product context:",
        f"Product name: {product_name}",
        "Use this only as reference context.",
        "",
        "Packaging context:",
        "The item may be inside any of these packaging types:",
        "- original retail packaging",
        "- generic brown shipping box",
        "- padded envelope or mailer",
        "- plastic bag or poly bag",
        "",
        "Important packaging rule:",
        "You are evaluating the visible condition of the package or item that is shown.",
        "Do NOT require the actual product itself to be visible in order to classify as good.",
        "If the item is inside packaging and the visible packaging appears intact and acceptable, it may still be classified as good.",
        "",
        "Inspection goal:",
        "Classify the visible condition into exactly one final result:",
        "- good",
        "- damaged",
        "- manual_review",
        "",
        "Decision definitions:",
        "",
        "1) good",
        "Choose good when the visible package or visible item appears intact and there is no meaningful visible damage.",
        "Good includes cases where:",
        "- the original retail box looks intact and acceptable",
        "- a generic brown box looks intact and acceptable",
        "- a padded envelope or bag looks closed and not materially damaged",
        "- only minor cosmetic wear is visible, such as light scuffs, small wrinkles, light corner wear, or normal handling wear",
        "- shrink wrap has light wrinkles, minor seam fraying, or small cosmetic irregularities",
        "- shrink wrap has a small cut, small nick, or slight opening in the wrap, as long as the overall package still appears intact and there is no major damage",
        "",
        "Acceptable packaging examples that should still be classified as good include:",
        "- light wrinkles or ripples in shrink wrap",
        "- minor fraying or roughness along a wrap seam",
        "- a small cut or small slit in shrink wrap",
        "- a slightly imperfect or slightly opened wrap seam",
        "- light corner wear on cardboard",
        "- small superficial scuffs or handling marks",
        "- minor edge wear that does not materially damage the package",
        "",
        "Treat wrap issues as cosmetic and still classify as good when ALL of the following are true:",
        "- the package still appears generally intact",
        "- contents are not exposed",
        "- there is no large tear or large hole",
        "- there is no major puncture or major opening",
        "- there is no crushing, collapse, or structural failure",
        "",
        "Do NOT reject as damaged only because:",
        "- the actual product is not visible",
        "- the item is inside packaging",
        "- there are small superficial wrinkles, light scuffs, or normal handling marks",
        "- the shrink wrap has a small cut, minor fraying, or a slightly imperfect seam",
        "",
        "2) damaged",
        "Choose damaged when there is clear visible evidence of meaningful package or item damage.",
        "Damaged includes visible signs such as:",
        "- a big rip, large tear, or large split in the packaging",
        "- a big hole or major puncture in the packaging",
        "- exposed contents when they should be enclosed",
        "- crushed, collapsed, heavily dented, or severely deformed box structure",
        "- major contamination, liquid exposure, heavy residue, or burn/scorch marks",
        "- heavy abrasions, major gouges, severe corner crushing, or structural failure",
        "- packaging that is clearly badly compromised, badly opened, or badly torn",
        "",
        "Important damaged rule:",
        "A small cut in wrap alone is NOT damaged.",
        "Minor wrap seam wear alone is NOT damaged.",
        "Use damaged only for clearly meaningful visible damage, such as big tears, big holes, exposed contents, or major structural damage.",
        "",
        "3) manual_review",
        "Choose manual_review whenever the evidence is insufficient, ambiguous, conflicting, or borderline.",
        "Manual_review should be used when:",
        "- images are blurry, dark, overexposed, or incomplete",
        "- packaging condition cannot be judged confidently from the angles shown",
        "- a possible defect is visible but not certain",
        "- the severity of visible wear is borderline between acceptable and clearly damaged",
        "- you cannot tell whether a mark is real damage, reflection, shadow, print pattern, or image artifact",
        "",
        "Do NOT choose manual_review for minor shrink-wrap wrinkles, seam fraying, small cuts, or light cosmetic wear alone when the package still appears intact and contents are not exposed.",
        "Use manual_review only when it is genuinely unclear whether the packaging is actually badly torn, badly opened, severely damaged, or exposing contents.",
        "",
        "Important decision rules:",
        "- Be conservative, but do not over-escalate minor cosmetic wrap issues.",
        "- When uncertain about a serious defect, choose manual_review.",
        "- Base the decision only on what is visible in the provided images.",
        "- Consider all images together before deciding.",
        "- Do not require the product itself to be visible if the inspection task is clearly about the visible package condition.",
        "",
        "Reasoning guidance:",
        "- Minor cosmetic wear on packaging alone does not require damaged.",
        "- A visible seam line, seam wrinkle, seam fraying, or small cut in wrap is not by itself evidence of meaningful damage.",
        "- If the package still appears intact and contents are not exposed, classify minor wrap wear as good rather than manual_review.",
        "- Clear big tears, big holes, exposed contents, crushing, or major structural damage should result in damaged.",
        "- Lack of visual certainty about major damage should result in manual_review.",
        "",
        "Return JSON only.",
        "Your output must match the required schema exactly.",
        "",
        "When filling the JSON:",
        "- result must be one of: good, damaged, manual_review",
        "- confidence must be between 0 and 1",
        "- reasons should be short and concrete",
        "- observed_defects should list only visible issues actually supported by the images",
        "- summary should be a short inspection conclusion",
    ])


def optimize_image(input_path, output_dir, request_id, max_size, quality):
    """Write a smaller, EXIF-normalized JPEG copy for the API and return its path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}-{request_id}-opt.jpg"

    with Image.open(input_path) as img:
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")
        img.thumbnail((max_size, max_size))
        img.save(output_path, "JPEG", quality=quality, optimize=True)

    return output_path


def image_to_data_url(image_path):
    ext = image_path.suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext == ".png":
        mime = "image/png"
    elif ext == ".webp":
        mime = "image/webp"
    else:
        raise ValueError(f"Unsupported image type: {image_path}")

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def extract_output_text(response_json):
    output_text = response_json.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = response_json.get("output")
    if isinstance(output, list):
        for item in output:
            content = item.get("content") if isinstance(item, dict) else None
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "output_text":
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        return text

    return None


def call_openai(api_key, model, openai_url, product_name, image_paths, timeout_s):
    content = [{"type": "input_text", "text": build_prompt(product_name)}]

    for image_path in image_paths:
        content.append({
            "type": "input_image",
            "image_url": image_to_data_url(image_path),
            "detail": "high",
        })

    body = {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "inspection_result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "result": {"type": "string", "enum": ["good", "damaged", "manual_review"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "reasons": {"type": "array", "items": {"type": "string"}},
                        "observed_defects": {"type": "array", "items": {"type": "string"}},
                        "summary": {"type": "string"},
                    },
                    "required": ["result", "confidence", "reasons", "observed_defects", "summary"],
                },
            }
        },
    }

    response = requests.post(
        openai_url,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        json=body,
        timeout=timeout_s,
    )

    data = response.json()

    if not response.ok:
        raise RuntimeError(f"OpenAI API error: {json.dumps(data, indent=2)}")

    raw_text = extract_output_text(data)
    if not raw_text:
        raise RuntimeError(f"No output text found: {json.dumps(data, indent=2)}")

    return {
        "inspection": json.loads(raw_text),
        "usage": data.get("usage"),
        "raw_response_id": data.get("id"),
    }


def run_inspection(cfg, image_paths, product_name, request_id, optimized_dir=None):
    """Optimize the given images and run one OpenAI inspection over all of them.

    Returns a dict with the verdict fields plus bookkeeping:
        result, confidence, summary, reasons, observed_defects,
        usage, raw_response_id, optimized_image_paths
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    resolved_paths = [Path(p).expanduser().resolve() for p in image_paths]
    for image_path in resolved_paths:
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

    if optimized_dir is None:
        optimized_dir = Path(cfg.save_dir) / "optimized"

    optimized_paths = [
        optimize_image(
            input_path=image_path,
            output_dir=optimized_dir,
            request_id=request_id,
            max_size=cfg.max_size,
            quality=cfg.quality,
        )
        for image_path in resolved_paths
    ]

    openai_result = call_openai(
        api_key=api_key,
        model=cfg.model,
        openai_url=cfg.openai_url,
        product_name=product_name,
        image_paths=optimized_paths,
        timeout_s=cfg.timeout_s,
    )

    inspection = openai_result["inspection"]
    return {
        "result": inspection.get("result", ""),
        "confidence": float(inspection.get("confidence", 0.0)),
        "summary": inspection.get("summary", ""),
        "reasons": inspection.get("reasons", []),
        "observed_defects": inspection.get("observed_defects", []),
        "usage": openai_result["usage"],
        "raw_response_id": openai_result["raw_response_id"],
        "optimized_image_paths": [str(p) for p in optimized_paths],
    }
