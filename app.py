"""
AI-Powered Precision Weed Management
Research prototype — Technical University of Munich.

Entry point for the Streamlit dashboard. UI composition lives here; model
loading, inference, rendering and the FarmBot boundary live in utils/.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

from utils import config as cfg
from utils import farmbot, model_manager, ui
from utils.dataset import class_mismatch_warning, load_dataset_config, resolve_class_names
from utils.detection import InferenceError, detections_to_rows, open_image, run_detection
from utils.visualization import annotate, density_overlay, placeholder_frame, to_png_bytes

PAGES = [
    "Overview",
    "Live monitoring",
    "Weed detection",
    "Autonomous weeding",
    "Model information",
    "Dataset",
    "About",
]

st.set_page_config(
    page_title="AI-Powered Precision Weed Management",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_dataset_config():
    return load_dataset_config()


@st.cache_data(show_spinner=False)
def get_benchmarks():
    return cfg.load_benchmarks()


@st.cache_data(show_spinner=False)
def get_monitoring_frame(path_str: str, fingerprint: str):
    try:
        image = Image.open(path_str)
        image.load()
        return image.convert("RGB"), ""
    except Exception as exc:
        return (
            placeholder_frame(message="Monitoring frame could not be loaded"),
            f"{Path(path_str).name} could not be opened: {exc}",
        )


def load_monitoring_frame():
    path = cfg.MONITORING_FRAME
    if not path.is_file():
        return (
            placeholder_frame(message="Add assets/dummy_monitoring.jpg to enable this panel"),
            f"No monitoring frame found at {path}.",
        )
    stat = path.stat()
    return get_monitoring_frame(str(path), f"{stat.st_size}-{int(stat.st_mtime)}")


def init_state():
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("last_annotated", None)
    st.session_state.setdefault("last_original", None)
    st.session_state.setdefault("result_origin", "")
    available = model_manager.available_model_keys()
    default = cfg.DEFAULT_MODEL_KEY if cfg.DEFAULT_MODEL_KEY in available else (
        available[0] if available else cfg.DEFAULT_MODEL_KEY
    )
    st.session_state.setdefault("active_model", default)


# --------------------------------------------------------------------------
# Shared chrome
# --------------------------------------------------------------------------

def render_sidebar(statuses):
    with st.sidebar:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:1.4rem;">
                <div style="width:30px;height:30px;border-radius:7px;background:{cfg.PALETTE['crop']};
                     display:flex;align-items:center;justify-content:center;color:#fff;
                     font-weight:700;font-family:'Inter Tight',sans-serif;">W</div>
                <div style="line-height:1.2;">
                    <div style="font-weight:600;font-size:0.95rem;">Weed Management</div>
                    <div style="font-size:0.75rem;color:{cfg.PALETTE['muted']};">TUM research prototype</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page = st.radio("Section", PAGES, label_visibility="collapsed")

        st.divider()
        st.markdown("**System status**")
        installed = [s for s in statuses if s.available]
        engine_ready = model_manager.ultralytics_available()

        st.markdown(
            f"""
            <div style="font-size:0.83rem;line-height:1.9;color:{cfg.PALETTE['muted']};">
                <div>Detection engine · <strong style="color:{cfg.PALETTE['crop'] if engine_ready else cfg.PALETTE['warning']};">
                    {'Ready' if engine_ready else 'Not installed'}</strong></div>
                <div>Models installed · <strong style="color:{cfg.PALETTE['ink']};">
                    {len(installed)} of {len(statuses)}</strong></div>
                <div>FarmBot actuator · <strong style="color:{cfg.PALETTE['warning']};">Not connected</strong></div>
                <div>Camera feed · <strong style="color:{cfg.PALETTE['warning']};">Static frame</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.caption(
            "Detection counts, confidences and timings shown in this dashboard "
            "are measured from the model you run. Published accuracy figures are "
            "labelled as such wherever they appear."
        )
    return page


def render_metric_strip():
    result = st.session_state.get("last_result")
    if result is None:
        metrics = [
            {"label": "Soybean plants", "value": "—", "swatch": "crop", "foot": "Awaiting a detection run"},
            {"label": "Weeds", "value": "—", "swatch": "weed", "foot": "Awaiting a detection run"},
            {"label": "Mean confidence", "value": "—", "foot": "Across all detected objects"},
            {"label": "Inference time", "value": "—", "foot": "Forward pass on CPU"},
        ]
    else:
        origin = st.session_state.get("result_origin", "last run")
        metrics = [
            {
                "label": "Soybean plants",
                "value": f"{result.crop_count}",
                "swatch": "crop",
                "foot": origin,
            },
            {
                "label": "Weeds",
                "value": f"{result.weed_count}",
                "swatch": "weed",
                "foot": f"{result.weed_share:.0f}% of detected objects",
            },
            {
                "label": "Mean confidence",
                "value": f"{result.mean_confidence * 100:.1f}",
                "unit": "%",
                "foot": f"Threshold {result.confidence_threshold:.2f}",
            },
            {
                "label": "Inference time",
                "value": f"{result.inference_ms:.0f}",
                "unit": "ms",
                "foot": f"{result.model_label} · CPU",
            },
        ]
    st.markdown(ui.metric_strip(metrics), unsafe_allow_html=True)


def model_selector(statuses, key: str, label: str = "Detection model"):
    """Render the model dropdown and keep the shared selection in sync."""
    available = [s for s in statuses if s.available]
    if not available:
        st.error(
            "No model checkpoints found. Copy the trained `.pt` files into "
            f"`{cfg.MODELS_DIR}` and reload the page."
        )
        return None

    keys = [s.spec.key for s in available]
    active = st.session_state.get("active_model")
    index = keys.index(active) if active in keys else 0

    chosen = st.selectbox(
        label,
        options=keys,
        index=index,
        key=key,
        format_func=lambda k: cfg.MODELS_BY_KEY[k].label,
    )
    st.session_state["active_model"] = chosen
    return chosen


def store_result(result, annotated, original, origin: str):
    st.session_state["last_result"] = result
    st.session_state["last_annotated"] = annotated
    st.session_state["last_original"] = original
    st.session_state["result_origin"] = origin


def execute_detection(image, model_key, confidence, iou, imgsz, source_name):
    """Load the model, run one inference pass and annotate the frame."""
    dataset = get_dataset_config()
    spec = cfg.MODELS_BY_KEY[model_key]

    with st.spinner(f"Running {spec.label} on CPU…"):
        model = model_manager.load_model(model_key)
        model_names = model_manager.model_class_names(model)
        names = resolve_class_names(dataset, model_names)
        result, prepared = run_detection(
            model,
            image,
            model_key=model_key,
            model_label=spec.label,
            class_names=names,
            confidence=confidence,
            iou=iou,
            imgsz=imgsz,
            source_name=source_name,
        )
        annotated = annotate(prepared, result.detections)

    warning = class_mismatch_warning(dataset, model_names)
    return result, annotated, prepared, warning


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------

def page_overview(statuses, benchmarks):
    st.markdown(
        ui.section_title(
            "What this system does",
            "The dashboard runs the detection stage of an autonomous weeding pipeline: an "
            "image of a soybean bed goes in, and a labelled map of crop and weed instances "
            "comes out, together with the statistics an intervention decision would rely on.",
        ),
        unsafe_allow_html=True,
    )

    steps = [
        {"name": "Capture", "detail": "FarmBot-mounted camera photographs the bed.", "state": ""},
        {"name": "Detect", "detail": "A lightweight YOLO model locates crop and weed instances.", "state": "active"},
        {"name": "Decide", "detail": "Weed count and confidence drive the intervention rule.", "state": "active"},
        {"name": "Act", "detail": "The gantry treats each weed. Hardware not connected.", "state": "blocked"},
    ]
    st.markdown(ui.pipeline(steps), unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown(
            ui.bullet_panel(
                "Start here",
                [
                    "Open Weed detection and upload a field image, or drag one onto the panel.",
                    "Pick one of the installed YOLO models and adjust the confidence threshold if needed.",
                    "Run the detection to get an annotated frame and a measured summary.",
                    "Open Autonomous weeding to see the intervention plan derived from that result.",
                ],
                intro="Four steps to a full pass through the pipeline.",
            ),
            unsafe_allow_html=True,
        )

        notes = benchmarks.study_notes or []
        if notes:
            st.markdown(
                ui.panel(
                    "Findings from the underlying study",
                    f'<p style="margin-bottom:0.6rem;">{ui.tag("Published study", "published")}</p>'
                    + "<ul>"
                    + "".join(f"<li>{note}</li>" for note in notes)
                    + "</ul>",
                ),
                unsafe_allow_html=True,
            )

    with right:
        installed = [s for s in statuses if s.available]
        missing = [s for s in statuses if not s.available]
        rows = [(s.spec.label, f"{s.size_mb} MB" if s.size_mb else "installed") for s in installed]
        rows += [(s.spec.label, "not installed") for s in missing]
        st.markdown(
            ui.panel(
                "Model availability",
                ui.kv_table(rows)
                + f'<p style="margin-top:0.7rem;font-size:0.82rem;">Checkpoint directory: '
                f"<code>{cfg.MODELS_DIR}</code></p>",
            ),
            unsafe_allow_html=True,
        )

        dataset = get_dataset_config()
        if dataset.loaded:
            class_rows = [(f"Class {i}", name) for i, name in sorted(dataset.names.items())]
            st.markdown(
                ui.panel("Classes in data.yaml", ui.kv_table(class_rows)),
                unsafe_allow_html=True,
            )
        else:
            st.warning(dataset.error or "data.yaml could not be read.")


def page_live_monitoring(statuses):
    frame, frame_error = load_monitoring_frame()

    st.markdown(
        ui.section_title(
            "Field monitoring",
            "This panel shows a stored frame from the FarmBot camera position. No live "
            "stream is connected — the frame does not change on its own. Running detection "
            "here uses the same pipeline the live feed will use once the camera is wired in.",
        ),
        unsafe_allow_html=True,
    )
    if frame_error:
        st.info(frame_error)

    left, right = st.columns([3, 2], gap="large")

    with left:
        result = st.session_state.get("last_result")
        monitored = (
            st.session_state.get("last_annotated")
            if st.session_state.get("result_origin", "").startswith("Monitoring")
            else None
        )
        st.image(
            monitored if monitored is not None else frame,
            use_container_width=True,
            caption="Bed 3 · camera position A · stored frame",
        )
        if monitored is not None:
            st.markdown(ui.legend(), unsafe_allow_html=True)

    with right:
        st.markdown(
            f"""
            <div class="wd-panel">
                <h3>Feed status</h3>
                {ui.kv_table([
                    ("Camera", "Sony IMX317 (stored frame)"),
                    ("Stream", "Not connected"),
                    ("Source", cfg.MONITORING_FRAME.name),
                    ("Resolution", f"{frame.size[0]} × {frame.size[1]} px"),
                    ("Frame captured", "Static asset"),
                    ("Dashboard time", datetime.now().strftime("%d %b %Y, %H:%M:%S")),
                ])}
            </div>
            """,
            unsafe_allow_html=True,
        )

        model_key = model_selector(statuses, key="monitor_model", label="Model for this frame")
        run = st.button(
            "Analyse current frame",
            type="primary",
            use_container_width=True,
            disabled=model_key is None,
        )

        origin_is_monitor = st.session_state.get("result_origin", "").startswith("Monitoring")
        if origin_is_monitor and st.session_state.get("last_result") is not None:
            res = st.session_state["last_result"]
            st.markdown(
                ui.panel(
                    "Last frame analysis",
                    f'<p style="margin-bottom:0.6rem;">{ui.tag("Measured on this frame")}</p>'
                    + ui.kv_table(
                        [
                            ("Model", res.model_label),
                            ("Weeds", str(res.weed_count)),
                            ("Soybean plants", str(res.crop_count)),
                            ("Mean confidence", f"{res.mean_confidence * 100:.1f}%"),
                            ("Inference time", f"{res.inference_ms:.0f} ms"),
                            ("Analysed at", datetime.fromtimestamp(res.timestamp).strftime("%H:%M:%S")),
                        ]
                    ),
                ),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                ui.panel(
                    "Frame analysis",
                    "<p>No detection has been run on this frame yet. "
                    "Analyse it to populate the counters above.</p>",
                ),
                unsafe_allow_html=True,
            )

    if run and model_key:
        try:
            result, annotated, prepared, warning = execute_detection(
                frame,
                model_key,
                cfg.DEFAULT_CONFIDENCE,
                cfg.DEFAULT_IOU,
                cfg.DEFAULT_IMGSZ,
                source_name=cfg.MONITORING_FRAME.name,
            )
        except (model_manager.ModelLoadError, InferenceError) as exc:
            st.error(str(exc))
        else:
            if warning:
                st.warning(warning)
            store_result(result, annotated, prepared, origin="Monitoring frame")
            st.rerun()


def page_weed_detection(statuses):
    st.markdown(
        ui.section_title(
            "Detect weeds in a field image",
            "Upload a photograph of a soybean bed. The selected model locates every crop "
            "and weed instance it recognises, and the summary is computed from that run.",
        ),
        unsafe_allow_html=True,
    )

    upload_col, control_col = st.columns([3, 2], gap="large")

    with upload_col:
        uploaded = st.file_uploader(
            "Field image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Drag a file here or browse. JPG, PNG, BMP and WEBP up to 20 MB.",
        )

    with control_col:
        model_key = model_selector(statuses, key="detect_model")
        if model_key:
            spec = cfg.MODELS_BY_KEY[model_key]
            st.caption(f"{spec.architecture} · checkpoint {spec.filename}")

        with st.expander("Detection settings"):
            confidence = st.slider(
                "Confidence threshold", 0.05, 0.95, cfg.DEFAULT_CONFIDENCE, 0.05,
                help="Objects below this score are discarded.",
            )
            iou = st.slider(
                "Overlap (IoU) threshold", 0.1, 0.9, cfg.DEFAULT_IOU, 0.05,
                help="How much two boxes may overlap before one is suppressed.",
            )
            imgsz = st.select_slider(
                "Inference resolution", options=[416, 512, 640, 768], value=cfg.DEFAULT_IMGSZ,
                help="Larger is more accurate on small weeds and slower on CPU.",
            )

    image = None
    if uploaded is not None:
        size_mb = getattr(uploaded, "size", 0) / (1024 * 1024)
        if size_mb > cfg.MAX_UPLOAD_MB:
            st.error(
                f"That file is {size_mb:.1f} MB. Upload an image under {cfg.MAX_UPLOAD_MB} MB."
            )
        else:
            try:
                image = open_image(uploaded)
            except InferenceError as exc:
                st.error(str(exc))

    run = st.button(
        "Run detection",
        type="primary",
        disabled=image is None or model_key is None,
        help=None if image is not None else "Upload an image first.",
    )

    if image is not None and st.session_state.get("result_origin") != f"Upload · {uploaded.name}":
        st.image(image, caption=f"{uploaded.name} · ready to analyse", width=520)

    if run and image is not None and model_key:
        try:
            result, annotated, prepared, warning = execute_detection(
                image, model_key, confidence, iou, imgsz, source_name=uploaded.name
            )
        except (model_manager.ModelLoadError, InferenceError) as exc:
            st.error(str(exc))
            return
        if warning:
            st.warning(warning)
        store_result(result, annotated, prepared, origin=f"Upload · {uploaded.name}")
        st.rerun()

    result = st.session_state.get("last_result")
    if result is None:
        st.markdown(
            ui.panel(
                "No results yet",
                "<p>Upload an image and run a detection. The annotated frame, the object "
                "table and the summary will appear here.</p>",
            ),
            unsafe_allow_html=True,
        )
        return

    render_results(result)


def render_results(result):
    annotated = st.session_state.get("last_annotated")
    original = st.session_state.get("last_original")

    st.markdown(
        ui.section_title(
            "Detection results",
            f"Source: {result.source_name or 'unnamed image'} · "
            f"{result.model_label} · {datetime.fromtimestamp(result.timestamp).strftime('%H:%M:%S')}",
        ),
        unsafe_allow_html=True,
    )

    if result.was_resized:
        st.caption(
            f"Image downscaled from {result.original_size[0]}×{result.original_size[1]} to "
            f"{result.image_size[0]}×{result.image_size[1]} px before inference."
        )

    image_col, summary_col = st.columns([3, 2], gap="large")

    with image_col:
        tab_detected, tab_original, tab_density = st.tabs(
            ["Detections", "Original image", "Weed density"]
        )
        with tab_detected:
            st.markdown(ui.legend(), unsafe_allow_html=True)
            if annotated is not None:
                st.image(annotated, use_container_width=True)
                st.download_button(
                    "Download annotated image",
                    data=to_png_bytes(annotated),
                    file_name=f"detections_{result.model_key}.png",
                    mime="image/png",
                )
            if not result.detections:
                st.info(
                    "The model found no objects above the confidence threshold. "
                    "Lower the threshold or try a higher inference resolution."
                )
        with tab_original:
            if original is not None:
                st.image(original, use_container_width=True, caption="Image as passed to the model")
        with tab_density:
            if original is not None:
                st.image(
                    density_overlay(original, result.detections),
                    use_container_width=True,
                    caption="Weed concentration per grid cell in this image",
                )
                st.caption(
                    "Shading reflects weed detections in this frame only. Field-level "
                    "density mapping needs georeferenced captures."
                )

    with summary_col:
        st.markdown(
            ui.panel(
                "Detection summary",
                f'<p style="margin-bottom:0.7rem;">{ui.tag("Measured on your image")}</p>'
                + ui.kv_table(result.as_summary_rows()),
            ),
            unsafe_allow_html=True,
        )

        class_counts = result.class_counts()
        if class_counts:
            st.markdown(
                ui.panel(
                    "Objects per class",
                    ui.kv_table([(name, str(count)) for name, count in class_counts.items()]),
                ),
                unsafe_allow_html=True,
            )

    if result.detections:
        with st.expander(f"All {result.total} detected objects"):
            rows = detections_to_rows(result)
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.download_button(
                "Download detections as CSV",
                data=rows_to_csv(rows),
                file_name=f"detections_{result.model_key}.csv",
                mime="text/csv",
            )


def rows_to_csv(rows) -> str:
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def page_autonomous_weeding():
    client = farmbot.get_client()
    status = client.status()
    result = st.session_state.get("last_result")

    st.markdown(
        ui.section_title(
            "Autonomous weeding",
            "The planning stages run on real detections. The actuator stage is inactive: "
            "no command is sent and no weeding takes place while the FarmBot is disconnected.",
        ),
        unsafe_allow_html=True,
    )

    detection_state = "active" if result else ""
    steps = [
        {
            "name": "Detection",
            "detail": (
                f"{result.total} objects from {result.model_label}." if result
                else "Run a detection first."
            ),
            "state": detection_state,
        },
        {
            "name": "Weed localisation",
            "detail": (
                f"{result.weed_count} weed centroids in image coordinates." if result
                else "Waiting for detections."
            ),
            "state": detection_state,
        },
        {
            "name": "Intervention decision",
            "detail": "Rule applied to weed count and confidence." if result else "Waiting for detections.",
            "state": detection_state,
        },
        {
            "name": "FarmBot action",
            "detail": "Blocked — no device connected.",
            "state": "blocked",
        },
    ]
    st.markdown(ui.pipeline(steps), unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")

    with right:
        st.markdown(
            ui.panel(
                "Actuator status",
                f'<p style="margin-bottom:0.7rem;">{ui.tag("Prototype mode", "prototype")}</p>'
                + ui.kv_table(
                    [
                        ("Connection", "Not connected"),
                        ("Mode", status.mode),
                        ("Device", status.device_name or "None configured"),
                        ("Firmware", status.firmware or "Unknown"),
                        ("Weeding passes run", "0"),
                    ]
                )
                + f'<p style="margin-top:0.7rem;">{status.detail}</p>',
            ),
            unsafe_allow_html=True,
        )

    with left:
        if result is None:
            st.markdown(
                ui.panel(
                    "No detection to plan from",
                    "<p>Run a detection on the Weed detection or Live monitoring page. "
                    "The intervention plan is built from that result.</p>",
                ),
                unsafe_allow_html=True,
            )
            return

        min_conf = st.slider(
            "Minimum confidence to treat a weed", 0.1, 0.95, 0.5, 0.05,
            help="Weeds below this score are left alone.",
        )
        min_weeds = st.number_input(
            "Weeds needed before a pass is queued", min_value=1, max_value=50, value=1, step=1
        )

        should_act, reason = farmbot.intervention_decision(
            result, weed_threshold=int(min_weeds), confidence_threshold=min_conf
        )
        (st.success if should_act else st.info)(reason)

        targets = farmbot.weed_targets(result, min_confidence=min_conf)
        annotated = st.session_state.get("last_annotated")
        original = st.session_state.get("last_original")
        if original is not None:
            st.image(
                annotate(
                    original,
                    [d for d in result.weeds if d.confidence >= min_conf],
                    mark_weed_centres=True,
                ),
                use_container_width=True,
                caption="Treatment targets with centroid markers",
            )

        if targets:
            rows = [
                {
                    "Target": t.label,
                    "Confidence": round(t.confidence * 100, 1),
                    "x (px)": t.pixel_x,
                    "y (px)": t.pixel_y,
                    "x (normalised)": t.norm_x,
                    "y (normalised)": t.norm_y,
                }
                for t in targets
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.caption(
                "Coordinates are in image space. Converting them to gantry coordinates "
                "requires the camera calibration and bed geometry from the FarmBot setup."
            )
            st.download_button(
                "Download target list",
                data=rows_to_csv(rows),
                file_name="weed_targets.csv",
                mime="text/csv",
            )

        if st.button("Send weeding job to FarmBot", disabled=not targets):
            try:
                client.run_weeding_sequence(targets)
            except farmbot.FarmBotNotConnected as exc:
                st.error(
                    f"{exc} The plan above is ready to send once a device is configured "
                    "in utils/farmbot.py."
                )


def page_model_information(statuses, benchmarks):
    st.markdown(
        ui.section_title(
            "Detection models",
            "Five nano-scale YOLO generations were compared for this task. Nano variants "
            "were chosen because inference has to run on the modest compute available at "
            "the FarmBot, without a GPU.",
        ),
        unsafe_allow_html=True,
    )

    if benchmarks.error:
        st.info(benchmarks.error + " Published accuracy figures are unavailable.")

    active = st.session_state.get("active_model")
    columns = st.columns(len(cfg.MODEL_REGISTRY), gap="small")
    for column, spec in zip(columns, cfg.MODEL_REGISTRY):
        benchmark = benchmarks.get(spec.key)
        value = f"{benchmark.map50:.4f}" if benchmark.map50 is not None else None
        with column:
            st.markdown(
                ui.model_card(
                    name=spec.label,
                    year=spec.released,
                    architecture=spec.architecture,
                    metric_label="mAP@0.5",
                    metric_value=value,
                    selected=spec.key == active,
                ),
                unsafe_allow_html=True,
            )

    st.markdown(
        f'<div style="margin:1.2rem 0 0.6rem 0;">{ui.tag("Published study", "published")} '
        f'<span style="color:{cfg.PALETTE["muted"]};font-size:0.88rem;margin-left:0.4rem;">'
        "Values below come from the training study, not from images you upload."
        "</span></div>",
        unsafe_allow_html=True,
    )

    rows = []
    for spec in cfg.MODEL_REGISTRY:
        benchmark = benchmarks.get(spec.key)
        rows.append(
            {
                "Model": spec.label,
                "Precision": f"{benchmark.precision:.4f}" if benchmark.precision is not None else "—",
                "Recall": f"{benchmark.recall:.4f}" if benchmark.recall is not None else "—",
                "mAP@0.5": f"{benchmark.map50:.4f}" if benchmark.map50 is not None else "—",
                "mAP@0.5:0.95": f"{benchmark.map50_95:.4f}" if benchmark.map50_95 is not None else "—",
                "Installed": "Yes" if spec.is_available else "No",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption(
        "A dash means the value is not recorded in benchmarks.json. Add measured "
        "figures there rather than to the interface, so published and observed "
        "numbers never get mixed up."
    )

    left, right = st.columns(2, gap="large")
    with left:
        if benchmarks.study_notes:
            st.markdown(
                ui.bullet_panel("Study findings", benchmarks.study_notes),
                unsafe_allow_html=True,
            )
    with right:
        st.markdown(
            ui.panel(
                "Installed checkpoints",
                ui.kv_table(
                    [
                        (
                            s.spec.label,
                            f"{s.size_mb} MB" if s.available and s.size_mb else
                            ("installed" if s.available else "not installed"),
                        )
                        for s in statuses
                    ]
                )
                + '<p style="margin-top:0.7rem;font-size:0.82rem;">Models load on first use '
                "and stay cached, so only the ones you actually run occupy memory.</p>",
            ),
            unsafe_allow_html=True,
        )


def page_dataset():
    dataset = get_dataset_config()

    st.markdown(
        ui.section_title(
            "Research dataset",
            "Images were captured in a soybean bed by the camera mounted on the FarmBot "
            "gantry and annotated with bounding boxes.",
        ),
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(
            ui.panel(
                "Collection and annotation",
                ui.kv_table(
                    [
                        ("Domain", "Soybean cultivation"),
                        ("Capture platform", "FarmBot-mounted camera"),
                        ("Sensor", "Sony IMX317"),
                        ("Annotation", "Bounding boxes"),
                        ("Annotation platform", "Roboflow"),
                        ("Curated images", "≈ 641"),
                    ]
                ),
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            ui.bullet_panel(
                "Augmentation",
                [
                    "Fog — reduced contrast and scattered light.",
                    "Night — low illumination and shifted colour balance.",
                    "High irradiance — strong direct sunlight and blown highlights.",
                ],
                intro=(
                    "Photometric transforms simulate conditions the gantry meets in the "
                    "field, so the model is not tuned to fair weather alone."
                ),
            ),
            unsafe_allow_html=True,
        )

    with right:
        if dataset.loaded:
            st.markdown(
                ui.panel(
                    "Classes from data.yaml",
                    ui.kv_table(
                        [(f"Index {i}", name) for i, name in sorted(dataset.names.items())]
                    )
                    + f'<p style="margin-top:0.7rem;font-size:0.82rem;">Read from '
                    f"<code>{dataset.path}</code></p>",
                ),
                unsafe_allow_html=True,
            )
            mapping_rows = [
                (name, "Weed" if name in dataset.weed_classes else "Soybean plant")
                for name in dataset.class_list
            ]
            st.markdown(
                ui.panel(
                    "How classes are grouped",
                    ui.kv_table(mapping_rows)
                    + '<p style="margin-top:0.7rem;font-size:0.82rem;">Grouping rules live in '
                    "<code>utils/config.py</code>. Adjust the keyword lists if your class "
                    "names differ.</p>",
                ),
                unsafe_allow_html=True,
            )
        else:
            st.warning(dataset.error or "data.yaml is not available.")
            st.markdown(
                ui.panel(
                    "Expected file",
                    "<p>Place the <code>data.yaml</code> used for training in the project "
                    "root. The dashboard reads its <code>names</code> entry so the "
                    "interface uses the dataset's own class names.</p>",
                ),
                unsafe_allow_html=True,
            )

    st.markdown(
        ui.panel(
            "Timing of detection",
            f'<p style="margin-bottom:0.6rem;">{ui.tag("Published study", "published")}</p>'
            "<p>The study found the first ten days after sowing to be the decisive window: "
            "weeds and soybean seedlings are still separable by size and shape, and removing "
            "weeds before canopy closure prevents most of the competition for light and "
            "nutrients.</p>",
        ),
        unsafe_allow_html=True,
    )


def page_about(benchmarks):
    st.markdown(
        ui.section_title(
            "About this prototype",
            "A research demonstrator built at the Technical University of Munich for "
            "AI-assisted weed management in soybean cultivation.",
        ),
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(
            ui.panel(
                "Research context",
                "<p>Weed competition reduces soybean yield, and the usual answers — broad "
                "chemical application or manual removal — are costly and environmentally "
                "blunt. Detecting individual weeds makes intervention selective: treat the "
                "weed, leave the crop and the soil around it alone.</p>"
                "<p style='margin-top:0.7rem;'>This dashboard covers the perception half of "
                "that loop and prepares the input the actuation half will need.</p>",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            ui.bullet_panel(
                "How numbers are sourced",
                [
                    "Detection counts, confidences and timings are measured from the model run you trigger.",
                    "Accuracy figures such as mAP@0.5 come from benchmarks.json and are labelled as published results.",
                    "Nothing on the actuator page implies that physical weeding occurred.",
                ],
                intro="Two kinds of numbers appear in this interface and are never mixed.",
            ),
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            ui.panel(
                "Built with",
                ui.kv_table(
                    [
                        ("Interface", "Streamlit"),
                        ("Detection", "Ultralytics YOLO"),
                        ("Rendering", "Pillow"),
                        ("Runtime", "CPU-only, free-tier friendly"),
                        ("Platform", "FarmBot open-source gantry"),
                    ]
                ),
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            ui.bullet_panel(
                "Planned extensions",
                [
                    "Live FarmBot camera feed replacing the stored frame.",
                    "FarmBot Python API for gantry movement and tool control.",
                    "Camera calibration to turn pixel centroids into bed coordinates.",
                    "Detection history and field-level weed-density maps.",
                ],
                intro="Each seam is already isolated in its own module.",
            ),
            unsafe_allow_html=True,
        )

    if benchmarks.citation:
        st.caption(benchmarks.citation)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main():
    st.markdown(ui.STYLESHEET, unsafe_allow_html=True)
    init_state()

    statuses = model_manager.inspect_models()
    benchmarks = get_benchmarks()

    page = render_sidebar(statuses)

    engine_ready = model_manager.ultralytics_available()
    installed = any(s.available for s in statuses)
    if engine_ready and installed:
        status_text, tone = "System online", "ok"
    elif not engine_ready:
        status_text, tone = "Detection engine unavailable", "warn"
    else:
        status_text, tone = "No models installed", "warn"

    st.markdown(
        ui.header(cfg.APP_TITLE, cfg.APP_SUBTITLE, cfg.INSTITUTION, status_text, tone),
        unsafe_allow_html=True,
    )
    render_metric_strip()

    if not engine_ready:
        st.error(
            "The Ultralytics package is not installed, so detection cannot run. "
            "Install the dependencies with `pip install -r requirements.txt` and restart."
        )

    if page == "Overview":
        page_overview(statuses, benchmarks)
    elif page == "Live monitoring":
        page_live_monitoring(statuses)
    elif page == "Weed detection":
        page_weed_detection(statuses)
    elif page == "Autonomous weeding":
        page_autonomous_weeding()
    elif page == "Model information":
        page_model_information(statuses, benchmarks)
    elif page == "Dataset":
        page_dataset()
    else:
        page_about(benchmarks)


if __name__ == "__main__":
    main()
