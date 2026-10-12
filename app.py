import os
import time
import io
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as tv_models
from PIL import Image

try:
    from rembg import remove, new_session
except Exception:
    remove = None
    new_session = None

try:
    import psutil
except Exception:
    psutil = None


st.set_page_config(page_title="Authentic vs Counterfeit Watch Detector", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


MODEL_CONFIGS = {
    "ResNet50": {
        "checkpoint": "outputs/resnet50/final_model/resnet50_final.pt",
        "eager_checkpoint": "outputs/resnet50/final_model/resnet50_final.pth",
        "default_threshold": 0.4149,
        "test_metrics": "outputs/resnet50/tables/table_final_model_resnet50.csv",
        "complexity": "outputs/resnet50/tables/complexity_resnet50.csv",
    },
    "EfficientNet-B0": {
        "checkpoint": "outputs/efficientnet_b0/final_model/efficientnet_b0_final.pt",
        "eager_checkpoint": "outputs/efficientnet_b0/final_model/efficientnet_b0_final.pth",
        "default_threshold": 0.3978,
        "test_metrics": "outputs/efficientnet_b0/tables/table_final_model_efficientnet_b0.csv",
        "complexity": "outputs/efficientnet_b0/tables/complexity_efficientnet_b0.csv",
    },
    "MobileNetV2": {
        "checkpoint": "outputs/mobilenet_v2/final_model/mobilenet_v2_final.pt",
        "eager_checkpoint": "outputs/mobilenet_v2/final_model/mobilenet_v2_final.pth",
        "default_threshold": 0.4244,
        "test_metrics": "outputs/mobilenet_v2/tables/table_final_model_mobilenet_v2.csv",
        "complexity": "outputs/mobilenet_v2/tables/complexity_mobilenet_v2.csv",
    },
}

BASE_DIR = Path(__file__).resolve().parent


def resolve_path(relative_path):
    return BASE_DIR / relative_path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_INPUT_SIZE = 320
TRIM_STRAP = True
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMGNET_MEAN, std=IMGNET_STD),
    ]
)


def _largest_connected_component(mask_uint8):
    mask_uint8 = np.asarray(mask_uint8, dtype=np.uint8)
    if mask_uint8.size == 0:
        return mask_uint8
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    if num_labels <= 1:
        return mask_uint8
    largest_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask_uint8)
    out[labels == largest_index] = 255
    return out


def _trim_strap_mask(mask_uint8):
    if not TRIM_STRAP:
        return mask_uint8
    mask_uint8 = np.asarray(mask_uint8, dtype=np.uint8)
    if mask_uint8.size == 0 or mask_uint8.shape[0] == 0 or mask_uint8.shape[1] == 0:
        return mask_uint8

    row_coverage = np.sum(mask_uint8 > 0, axis=1)
    if row_coverage.size == 0:
        return mask_uint8

    max_width = float(row_coverage.max())
    if max_width <= 0:
        return mask_uint8

    candidate_rows = np.where(row_coverage >= 0.7 * max_width)[0]
    if candidate_rows.size == 0:
        return mask_uint8

    center_row = int(np.median(candidate_rows))
    window_height = max(1, int(round(max_width)))
    top = max(0, center_row - window_height // 2)
    bottom = min(mask_uint8.shape[0], top + window_height)
    if bottom - top < max(1, int(round(0.5 * mask_uint8.shape[1]))):
        return mask_uint8

    trimmed = np.zeros_like(mask_uint8)
    trimmed[top:bottom, :] = mask_uint8[top:bottom, :]
    return trimmed


def _pad_to_square(image_obj, background_color=(255, 255, 255)):
    width, height = image_obj.size
    size = max(width, height)
    padded = Image.new("RGB", (size, size), color=background_color)
    offset_left = (size - width) // 2
    offset_top = (size - height) // 2
    padded.paste(image_obj, (offset_left, offset_top))
    return padded


@st.cache_resource
def get_rembg_session():
    return new_session() if new_session is not None else None


def notebook_preprocess_image(image, trim_strap=True):
    rgb_img = image.convert("RGB") if image.mode != "RGB" else image.copy()
    width, height = rgb_img.size
    if min(width, height) < 400:
        st.warning(
            "This image fails the notebook's 400 px short-side cleaning gate; it may already be processed or too small for this pipeline."
        )

    try:
        if remove is None:
            raise RuntimeError("rembg backend not available")
        removed = remove(rgb_img, session=get_rembg_session())
        if removed.mode in {"RGBA", "LA"}:
            alpha = np.asarray(removed)[:, :, 3]
            mask = (alpha > 0).astype(np.uint8) * 255
        else:
            arr = np.asarray(removed.convert("RGB"), dtype=np.uint8)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            mask = (gray > 10).astype(np.uint8) * 255
    except Exception:
        gray = cv2.cvtColor(np.asarray(rgb_img, dtype=np.uint8), cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        if np.all(mask == 0):
            mask = np.zeros((height, width), dtype=np.uint8)

    if mask.size == 0 or np.all(mask == 0):
        return rgb_img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.Resampling.BILINEAR)

    mask = _largest_connected_component(mask)
    if trim_strap:
        mask = _trim_strap_mask(mask)

    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return rgb_img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.Resampling.BILINEAR)

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    case_width = max(x1 - x0, y1 - y0)
    margin = max(1, int(round(0.04 * case_width)))
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(width, x1 + margin)
    y1 = min(height, y1 + margin)

    crop = rgb_img.crop((x0, y0, x1, y1))
    crop_arr = np.asarray(crop, dtype=np.uint8)
    crop_mask = mask[y0:y1, x0:x1]
    if crop_mask.size == 0:
        return rgb_img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.Resampling.BILINEAR)

    crop_mask = cv2.dilate(crop_mask.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1)
    masked = crop_arr.copy()
    fg = crop_mask > 0
    masked[~fg] = 255

    square_crop = _pad_to_square(Image.fromarray(masked, mode="RGB"), background_color=(255, 255, 255))
    square_crop = square_crop.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.Resampling.BILINEAR)
    if min(crop.size) < 320:
        return rgb_img.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.Resampling.BILINEAR)
    return square_crop


def sigmoid_scalar(x):
    return float(1.0 / (1.0 + np.exp(-x)))


def _get_out_features(state_dict, keys, default=1):
    for key in keys:
        if key in state_dict and hasattr(state_dict[key], "shape") and len(state_dict[key].shape) >= 1:
            return int(state_dict[key].shape[0])
    return int(default)


def build_eager_model(model_name, state_dict):
    if model_name == "ResNet50":
        model = tv_models.resnet50(weights=None)
        out_features = _get_out_features(state_dict, ["fc.1.weight", "fc.weight"], default=1)
        if "fc.1.weight" in state_dict:
            model.fc = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(model.fc.in_features, out_features),
            )
        else:
            model.fc = nn.Linear(model.fc.in_features, out_features)
        return model
    if model_name == "EfficientNet-B0":
        model = tv_models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        out_features = _get_out_features(state_dict, ["classifier.1.weight"], default=1)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, out_features),
        )
        return model
    if model_name == "MobileNetV2":
        model = tv_models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        out_features = _get_out_features(state_dict, ["classifier.1.weight"], default=1)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, out_features),
        )
        return model
    raise ValueError(f"Unsupported model name for eager load: {model_name}")


def normalize_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            state_dict = ckpt_obj["state_dict"]
        elif "model_state_dict" in ckpt_obj and isinstance(ckpt_obj["model_state_dict"], dict):
            state_dict = ckpt_obj["model_state_dict"]
        else:
            state_dict = ckpt_obj
    else:
        state_dict = ckpt_obj

    cleaned = {}
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith("module.") else k
        cleaned[new_key] = v
    return cleaned


def disable_inplace_ops(model):
    changed = 0
    for module in model.modules():
        if hasattr(module, "inplace") and getattr(module, "inplace") is True:
            module.inplace = False
            changed += 1
    return changed


@st.cache_resource
def load_models():
    loaded = {}
    missing = []
    issues = []
    for model_name, cfg in MODEL_CONFIGS.items():
        ckpt = resolve_path(cfg["checkpoint"])
        if not ckpt.exists():
            missing.append(str(ckpt))
            continue
        try:
            model = torch.load(str(ckpt), map_location=DEVICE, weights_only=False)
            if not isinstance(model, nn.Module):
                raise TypeError(f"Expected a saved torch.nn.Module, got {type(model).__name__}")
            model = model.to(DEVICE)
            model.eval()
            loaded[model_name] = model
        except Exception as exc:
            issues.append(f"[{model_name}] inference checkpoint load issue: {exc}")
    return loaded, missing, issues


@st.cache_resource
def load_eager_cam_models():
    eager_models = {}
    missing = []
    issues = []
    thresholds = {}

    for model_name, cfg in MODEL_CONFIGS.items():
        eager_path = resolve_path(cfg.get("eager_checkpoint", ""))
        if not eager_path.exists():
            missing.append(str(eager_path))
            continue

        try:
            ckpt_obj = torch.load(str(eager_path), map_location=DEVICE, weights_only=True)
            state_dict = normalize_state_dict(ckpt_obj)
            model = build_eager_model(model_name, state_dict).to(DEVICE)
            model.load_state_dict(state_dict, strict=True)

            # Some torch/torchvision combinations throw Grad-CAM backward-hook
            # RuntimeError with inplace activations; disable inplace to stabilize.
            disable_inplace_ops(model)
            model.eval()

            threshold = float(ckpt_obj["threshold"])
            if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError(f"Invalid decision threshold in {eager_path.name}: {threshold}")
            eager_models[model_name] = model
            thresholds[model_name] = threshold
        except Exception as exc:
            issues.append(f"[{model_name}] eager checkpoint load issue: {exc}")

    return eager_models, missing, issues, thresholds


@st.cache_data
def load_offline_metrics(checkpoint_thresholds):
    required_test_cols = {
        "threshold",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
    }
    required_complexity_cols = {
        "total_params_M",
        "flops_M_macs",
        "model_size_MB",
        "inference_ms_per_image",
        "train_min_final_model",
        "peak_vram_inference_MB",
    }

    rows = []
    threshold_by_model = dict(checkpoint_thresholds)
    issues = []

    for model_name, cfg in MODEL_CONFIGS.items():
        row = {"Model": model_name}

        try:
            test_path = resolve_path(cfg["test_metrics"])
            if not test_path.exists():
                raise FileNotFoundError(f"Missing metrics file: {test_path}")

            test_df = pd.read_csv(test_path)
            missing_test_cols = required_test_cols - set(test_df.columns)
            if missing_test_cols:
                raise ValueError(
                    f"Schema mismatch in {test_path.name}. Missing: {sorted(missing_test_cols)}"
                )

            if test_df.empty:
                raise ValueError(f"No final-model metrics found in {test_path.name}")
            selected = test_df.iloc[0]
            table_threshold = float(selected["threshold"])
            threshold_by_model.setdefault(model_name, table_threshold)

            row.update(
                {
                    "Threshold Used": threshold_by_model[model_name],
                    "Accuracy": float(selected["test_accuracy"]),
                    "Precision": float(selected["test_precision"]),
                    "Recall": float(selected["test_recall"]),
                    "F1-Score": float(selected["test_f1"]),
                    "ROC-AUC": float(selected["test_roc_auc"]),
                }
            )
        except (FileNotFoundError, ValueError, KeyError, IndexError, pd.errors.ParserError) as exc:
            issues.append(f"[{model_name}] test metrics issue: {exc}")
            threshold_by_model.setdefault(model_name, cfg["default_threshold"])
            row.update(
                {
                    "Threshold Used": threshold_by_model[model_name],
                    "Accuracy": np.nan,
                    "Precision": np.nan,
                    "Recall": np.nan,
                    "F1-Score": np.nan,
                    "ROC-AUC": np.nan,
                }
            )

        try:
            complexity_path = resolve_path(cfg["complexity"])
            if not complexity_path.exists():
                raise FileNotFoundError(f"Missing complexity file: {complexity_path}")

            complexity_df = pd.read_csv(complexity_path)
            missing_complex_cols = required_complexity_cols - set(complexity_df.columns)
            if missing_complex_cols:
                raise ValueError(
                    f"Schema mismatch in {complexity_path.name}. Missing: {sorted(missing_complex_cols)}"
                )

            if complexity_df.empty:
                raise ValueError(f"No complexity metrics found in {complexity_path.name}")
            complexity_row = complexity_df.iloc[0]
            row.update(
                {
                    "Total Params (M)": float(complexity_row["total_params_M"]),
                    "FLOPs (M)": float(complexity_row["flops_M_macs"]),
                    "Model Size (MB, CSV)": float(complexity_row["model_size_MB"]),
                    "Latency (ms/img, CSV)": float(complexity_row["inference_ms_per_image"]),
                    "Training Time (min, CSV)": float(complexity_row["train_min_final_model"]),
                    "Peak GPU VRAM (MB, CSV)": float(complexity_row["peak_vram_inference_MB"]),
                }
            )
        except (FileNotFoundError, ValueError, KeyError, IndexError, pd.errors.ParserError) as exc:
            issues.append(f"[{model_name}] complexity metrics issue: {exc}")
            row.update(
                {
                    "Total Params (M)": np.nan,
                    "FLOPs (M)": np.nan,
                    "Model Size (MB, CSV)": np.nan,
                    "Latency (ms/img, CSV)": np.nan,
                    "Training Time (min, CSV)": np.nan,
                    "Peak GPU VRAM (MB, CSV)": np.nan,
                }
            )

        ckpt_path = resolve_path(cfg["checkpoint"])
        row["Checkpoint Size (MB, Actual)"] = (
            ckpt_path.stat().st_size / (1024 * 1024) if ckpt_path.exists() else np.nan
        )

        rows.append(row)

    return pd.DataFrame(rows), threshold_by_model, issues


def process_rss_mb():
    if psutil is None:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def run_inference_with_empirical_stats(model, processed_image, threshold, repeats=5):
    tensor_img = TRANSFORM(processed_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        _ = model(tensor_img)

    latencies_ms = []
    peak_gpu_mb = None
    rss_samples_mb = []

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(DEVICE)

    baseline_rss = process_rss_mb()
    with torch.no_grad():
        for _ in range(repeats):
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            logit = model(tensor_img).item()
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

            rss_now = process_rss_mb()
            if rss_now is not None:
                rss_samples_mb.append(rss_now)

    if DEVICE.type == "cuda":
        peak_gpu_mb = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024)

    peak_rss_delta_mb = None
    if rss_samples_mb and baseline_rss is not None:
        peak_rss_delta_mb = max(rss_samples_mb) - baseline_rss

    mean_latency = float(np.mean(latencies_ms))
    std_latency = float(np.std(latencies_ms, ddof=1)) if len(latencies_ms) > 1 else 0.0
    ci95 = 1.96 * std_latency / np.sqrt(len(latencies_ms)) if len(latencies_ms) > 1 else 0.0

    prob_ai = sigmoid_scalar(logit)
    verdict = "Counterfeit" if prob_ai >= threshold else "Authentic Real"

    return {
        "verdict": verdict,
        "prob_ai": prob_ai,
        "threshold": threshold,
        "latency_mean_ms": mean_latency,
        "latency_std_ms": std_latency,
        "latency_ci95_ms": ci95,
        "peak_cpu_rss_delta_mb": peak_rss_delta_mb,
        "peak_gpu_mem_mb": peak_gpu_mb,
    }


def _is_conv_like(module):
    if isinstance(module, nn.Conv2d):
        return True
    name = module.__class__.__name__.lower()
    if "conv" in name:
        return True
    weight = getattr(module, "weight", None)
    return weight is not None and hasattr(weight, "dim") and weight.dim() == 4


def is_torchscript_model(model):
    # Grad-CAM relies on forward hooks, unsupported on ScriptModule.
    return "scriptmodule" in model.__class__.__name__.lower()


def get_target_layer_for_gradcam(model, model_name):
    # Prefer architecture-specific terminal conv blocks when available.
    lower_name = model_name.lower()
    try:
        if "resnet" in lower_name and hasattr(model, "layer4"):
            return model.layer4[-1]
        if "efficientnet" in lower_name and hasattr(model, "features"):
            return model.features[-1]
        if "mobilenet" in lower_name and hasattr(model, "features"):
            return model.features[-1]
    except Exception:
        pass

    # TorchScript can hide canonical attributes; fallback to last conv-like module.
    modules = list(model.named_modules())
    for _, module in reversed(modules):
        if _is_conv_like(module):
            return module
    return None


def render_heatmap_overlay(rgb_float, heat, intensity=0.75, opacity=0.55):
    intensity = float(np.clip(intensity, 0.0, 1.0))
    opacity = float(np.clip(opacity, 0.0, 1.0))
    gamma = 1.0 - 0.6 * intensity
    threshold = 0.62 - 0.22 * intensity
    boost = 0.02 + 0.14 * intensity

    heat = np.power(np.clip(heat, 0.0, 1.0), gamma)
    strong_mask = np.clip((heat - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)

    color = np.zeros_like(rgb_float)
    color[..., 0] = np.clip(1.35 * heat, 0.0, 1.0)  # red channel
    color[..., 1] = np.clip(1.15 * (1.0 - np.abs(2.0 * heat - 1.0)), 0.0, 1.0)  # yellow mid-tones
    color[..., 2] = np.clip(0.55 * (1.0 - heat), 0.0, 1.0)  # cool low-importance background

    blended = np.clip((1.0 - opacity) * rgb_float + opacity * color, 0.0, 1.0)
    blended = np.clip(
        blended + strong_mask[..., None] * np.array([boost, 0.35 * boost, 0.0], dtype=np.float32),
        0.0,
        1.0,
    )
    return (blended * 255.0).astype(np.uint8)


def compute_gradcam_map(model, model_name, pil_image):
    target_layer = get_target_layer_for_gradcam(model, model_name)
    if target_layer is None:
        raise RuntimeError(f"Could not locate a convolutional target layer for {model_name}.")

    input_img = pil_image.resize((224, 224), Image.BILINEAR)
    tensor_img = TRANSFORM(input_img).unsqueeze(0).to(DEVICE)

    activations = []
    gradients = []

    def _forward_hook(_module, _inputs, output):
        activations.append(output.detach())

    def _backward_hook(_module, _grad_inputs, grad_outputs):
        gradients.append(grad_outputs[0].detach())

    forward_handle = target_layer.register_forward_hook(_forward_hook)
    backward_handle = target_layer.register_full_backward_hook(_backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        pred = model(tensor_img)
        if pred.ndim == 2 and pred.shape[1] > 1:
            target_index = int(torch.argmax(pred, dim=1).item())
            target_score = pred[:, target_index].sum()
        else:
            target_score = pred.reshape(-1)[0]

        target_score.backward()

        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks captured no activations/gradients.")

        acts = activations[-1]
        grads = gradients[-1]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1))

        cam_np = cam.squeeze(0).detach().cpu().numpy().astype(np.float32)
        cam_np -= float(cam_np.min())
        max_val = float(cam_np.max())
        if max_val <= 1e-8:
            raise RuntimeError("Degenerate CAM heatmap (all zeros).")
        cam_np /= max_val

        cam_img = Image.fromarray((cam_np * 255.0).astype(np.uint8), mode="L")
        cam_img = cam_img.resize((224, 224), Image.BILINEAR)
        heat = np.asarray(cam_img).astype(np.float32) / 255.0
        return heat
    finally:
        forward_handle.remove()
        backward_handle.remove()


def compute_saliency_map(model, pil_image):
    input_img = pil_image.resize((224, 224), Image.BILINEAR)
    tensor_img = TRANSFORM(input_img).unsqueeze(0).to(DEVICE)
    tensor_img.requires_grad_(True)

    model.zero_grad(set_to_none=True)
    logit = model(tensor_img).squeeze()
    logit.backward()

    grad = tensor_img.grad.detach().abs().squeeze(0)
    saliency = grad.max(dim=0).values
    saliency = saliency - saliency.min()
    saliency = saliency / (saliency.max() + 1e-8)
    return saliency.cpu().numpy().astype(np.float32)


def compute_saliency_overlay(model, pil_image, intensity=0.75, opacity=0.55):
    input_img = pil_image.resize((224, 224), Image.BILINEAR)
    rgb_float = np.asarray(input_img).astype(np.float32) / 255.0
    heat = compute_saliency_map(model, pil_image)

    intensity = float(np.clip(intensity, 0.0, 1.0))
    # Keep saliency fallback slightly stronger so it remains interpretable vs Grad-CAM.
    return render_heatmap_overlay(rgb_float, heat, intensity=min(1.0, intensity + 0.05), opacity=opacity)


def format_exc(exc, max_len=180):
    msg = str(exc).strip().replace("\n", " ")
    if not msg:
        msg = "No error message was provided by the backend."
    return msg[:max_len] + ("..." if len(msg) > max_len else "")


st.title("Counterfeit vs Authentic Watch Detection")
st.caption("Comparative Performance Analysis of ResNet50, EfficientNet-B0, and MobileNetV2 for Detecting Counterfeit Analog Watch Images")

models, missing_checkpoints, model_load_issues = load_models()
eager_cam_models, missing_eager_checkpoints, eager_cam_issues, checkpoint_thresholds = load_eager_cam_models()
metrics_df, thresholds, metric_issues = load_offline_metrics(checkpoint_thresholds)

if missing_checkpoints:
    st.warning(
        "Missing inference checkpoints:\n- "
        + "\n- ".join(missing_checkpoints)
    )

if model_load_issues:
    st.error("\n".join(["Inference checkpoint loading issues:", *model_load_issues]))

if metric_issues:
    st.error(
        "\n".join([
            "Metrics loading issues:",
            *metric_issues,
        ])
    )

if missing_eager_checkpoints:
    st.warning(
        "Missing eager checkpoints for Grad-CAM:\n- "
        + "\n- ".join(missing_eager_checkpoints)
    )

if eager_cam_issues:
    st.error(
        "\n".join([
            "Eager Grad-CAM model loading issues:",
            *eager_cam_issues,
        ])
    )

st.markdown(f"**Runtime Device:** {DEVICE}")
col_a, col_b = st.columns([2, 1])
with col_a:
    uploaded_file = st.file_uploader(
        "Sample Image", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=False
    )
with col_b:
    repeats = st.slider(
        "Empirical timing repeats per model",
        min_value=3,
        max_value=20,
        value=15,
        step=1,
        help="Each model runs this many times; we report mean and std latency.",
    )

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    image_key = hashlib.sha1(image_bytes).hexdigest()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    st.image(image, caption="Uploaded image", width="stretch")

    if st.session_state.get("analysis_image_key") != image_key:
        st.session_state["analysis_image_key"] = image_key
        st.session_state.pop("analysis_result", None)

    if st.button("Run Counterfeit Detection Analysis", type="primary"):
        if not models:
            st.error("No models could be loaded. Check the checkpoint warnings above.")
        else:
            with st.spinner("Running inference and collecting empirical metrics..."):
                result_rows = []
                explainability_maps = {}
                processed_image = notebook_preprocess_image(image, trim_strap=TRIM_STRAP)
                for model_name, cfg in MODEL_CONFIGS.items():
                    model = models.get(model_name)
                    if model is None:
                        continue
                    inf = run_inference_with_empirical_stats(
                        model=model,
                        processed_image=processed_image,
                        threshold=thresholds.get(model_name, cfg["default_threshold"]),
                        repeats=repeats,
                    )
                    result_rows.append(
                        {
                            "Model": model_name,
                            "Verdict": inf["verdict"],
                            "AI Probability": inf["prob_ai"],
                            "Threshold Used": inf["threshold"],
                            "Empirical Latency Mean (ms/img)": inf["latency_mean_ms"],
                            "Empirical Latency Std (ms)": inf["latency_std_ms"],
                            "Empirical Latency 95% CI (ms)": inf["latency_ci95_ms"],
                            "Empirical Peak CPU RSS Delta (MB)": inf["peak_cpu_rss_delta_mb"],
                            "Empirical Peak GPU Mem (MB)": inf["peak_gpu_mem_mb"],
                        }
                    )

                    cam_model = eager_cam_models.get(model_name)
                    if cam_model is None:
                        explainability_maps[model_name] = {
                            "method": "Gradient Saliency (missing eager .pth for Grad-CAM)",
                            "heat": compute_saliency_map(model, image),
                        }
                    else:
                        try:
                            explainability_maps[model_name] = {
                                "method": "Grad-CAM",
                                "heat": compute_gradcam_map(cam_model, model_name, image),
                            }
                        except Exception as cam_exc:
                            explainability_maps[model_name] = {
                                "method": (
                                    f"Gradient Saliency (fallback: {type(cam_exc).__name__}: "
                                    f"{format_exc(cam_exc)})"
                                ),
                                "heat": compute_saliency_map(cam_model, image),
                            }

                live_df = pd.DataFrame(result_rows)
                final_df = live_df.merge(metrics_df, on="Model", how="left")
                st.session_state["analysis_result"] = {
                    "live_df": live_df,
                    "final_df": final_df,
                    "explainability_maps": explainability_maps,
                }

    analysis_result = st.session_state.get("analysis_result")
    if analysis_result is not None:
        live_df = analysis_result["live_df"]
        final_df = analysis_result["final_df"]
        explainability_maps = analysis_result["explainability_maps"]
        display_rgb_float = np.asarray(image.resize((224, 224), Image.BILINEAR)).astype(np.float32) / 255.0

        st.subheader("Per-Model Verdict")
        verdict_cols = st.columns(len(live_df))
        for idx, row in live_df.reset_index(drop=True).iterrows():
            with verdict_cols[idx]:
                st.markdown(f"### {row['Model']}")
                if row["Verdict"] == "Counterfeit":
                    st.error(f"{row['Verdict']}")
                else:
                    st.success(f"{row['Verdict']}")
                st.metric("Counterfeit Probability", f"{row['AI Probability'] * 100:.2f}%")
                st.metric("Decision Threshold", f"{row['Threshold Used']:.3f}")

        st.subheader("Model Comparison Table")
        st.dataframe(
            final_df,
            width="stretch",
            hide_index=True,
            column_config={
                "AI Probability": st.column_config.ProgressColumn(
                    "AI Probability", min_value=0.0, max_value=1.0, format="%.4f"
                )
            },
        )

        st.subheader("Model Quality Metrics (from exported CSV)")
        quality_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        st.dataframe(final_df[["Model"] + quality_cols], width="stretch", hide_index=True)

        st.subheader("Empirical Time and Space")
        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.markdown("Latency Mean (ms/img)")
            st.bar_chart(final_df.set_index("Model")["Empirical Latency Mean (ms/img)"])
        with chart_cols[1]:
            st.markdown("Checkpoint Size (MB, Actual)")
            st.bar_chart(final_df.set_index("Model")["Checkpoint Size (MB, Actual)"])

        if DEVICE.type == "cuda":
            st.markdown("Empirical Peak GPU Memory (MB)")
            st.bar_chart(final_df.set_index("Model")["Empirical Peak GPU Mem (MB)"])
        else:
            st.info(
                "CUDA not detected. GPU memory plot is unavailable for this session."
            )

        if psutil is None:
            st.warning(
                "psutil unavailable. CPU RSS delta metric is disabled in this session."
            )

        st.subheader("Explainability Heatmaps")
        st.caption(
            "Use the controls below to adjust overlay visibility instantly without rerunning inference. "
            "True Grad-CAM uses eager .pth checkpoints; fallback uses gradient saliency."
        )
        explain_col1, explain_col2 = st.columns([1, 1])
        with explain_col1:
            explainability_intensity = st.slider(
                "Explainability intensity",
                min_value=0.0,
                max_value=1.0,
                value=0.75,
                step=0.05,
                key="explainability_intensity_slider",
            )
        with explain_col2:
            explainability_opacity = st.slider(
                "Overlay opacity",
                min_value=0.05,
                max_value=0.95,
                value=0.55,
                step=0.05,
                key="explainability_opacity_slider",
            )

        saliency_cols = st.columns(len(live_df))
        for idx, row in live_df.reset_index(drop=True).iterrows():
            model_name = row["Model"]
            with saliency_cols[idx]:
                heat_item = explainability_maps.get(model_name)
                if heat_item is None:
                    current_model = models[model_name]
                    saliency_img = compute_saliency_overlay(
                        current_model,
                        image,
                        intensity=explainability_intensity,
                        opacity=explainability_opacity,
                    )
                    explanation_label = "Gradient Saliency"
                else:
                    saliency_img = render_heatmap_overlay(
                        display_rgb_float,
                        heat_item["heat"],
                        intensity=explainability_intensity,
                        opacity=explainability_opacity,
                    )
                    explanation_label = heat_item["method"]
                st.markdown(f"**{model_name}**")
                st.image(saliency_img, caption=explanation_label, width="stretch")
    else:
        st.info("Run Counterfeit Detection Analysis to generate predictions, metrics, and explainability maps.")

else:
    st.info("Awaiting sample image.")


with st.expander("Reference: Offline complexity/training metrics used in this app"):
    st.dataframe(metrics_df, width="stretch", hide_index=True)