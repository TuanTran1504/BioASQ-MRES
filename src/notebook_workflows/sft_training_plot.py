"""Publication-style SFT training comparison plots for notebook 11."""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median
from typing import Mapping

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DEFAULT_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    windows = Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf")
    candidates = [windows, Path("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")]
    for candidate in candidates:
        try:
            return ImageFont.truetype(str(candidate), size)
        except OSError:
            continue
    return ImageFont.load_default()


def _load_run(run: Path) -> dict:
    metrics_path = run / "adapter/training_metrics.csv"
    history_path = run / "generated_dev_selection/history.json"
    best_path = run / "generated_dev_selection/best_summary.json"
    for path in (metrics_path, history_path, best_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    history = json.loads(history_path.read_text(encoding="utf-8"))
    best = json.loads(best_path.read_text(encoding="utf-8"))
    return {
        "metrics": pd.read_csv(metrics_path),
        "history": history,
        "best": best,
        "run": run,
    }


def _lighten(color: str, fraction: float = 0.68) -> str:
    value = color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Expected a six-digit hex color, got {color!r}")
    channels = [int(value[index:index + 2], 16) for index in (0, 2, 4)]
    light = [round(channel + (255 - channel) * fraction) for channel in channels]
    return "#" + "".join(f"{channel:02X}" for channel in light)


def draw_sft_training_comparison(
    runs: Mapping[str, str | Path],
    output_path: str | Path,
    *,
    colors: Mapping[str, str] | None = None,
    title: str = "Supervised fine-tuning dynamics",
    subtitle: str = "Qwen2.5 factoid extractors | shared 160-question BioASQ development set",
    max_seq_length: int = 4096,
    font_scale: float = 1.30,
) -> dict:
    """Draw training loss, validation loss and generated-dev MRR for SFT runs."""
    if len(runs) != 2:
        raise ValueError("This three-panel layout currently expects exactly two SFT runs")
    if font_scale <= 0:
        raise ValueError("font_scale must be positive")
    resolved = {label: Path(path).resolve() for label, path in runs.items()}
    palette = dict(colors or {})
    for index, label in enumerate(resolved):
        palette.setdefault(label, DEFAULT_COLORS[index])
    pale = {label: _lighten(color) for label, color in palette.items()}

    loaded = {label: _load_run(path) for label, path in resolved.items()}
    width, height = 3300, 1200
    background, panel, ink = "#F7F8FA", "#FFFFFF", "#172033"
    muted, grid_color, border = "#5B6575", "#E6E9EF", "#D8DDE6"
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    scaled_font = lambda size, bold=False: _font(round(size * font_scale), bold=bold)
    fonts = {
        "title": scaled_font(56, bold=True), "subtitle": scaled_font(28),
        "panel_label": scaled_font(26, bold=True), "panel_title": scaled_font(33, bold=True),
        "axis": scaled_font(24), "tick": scaled_font(21),
        "note": scaled_font(20), "legend": scaled_font(24),
    }

    draw.text((100, 66), title, font=fonts["title"], fill=ink, anchor="la")
    draw.text((100, 135), subtitle, font=fonts["subtitle"], fill=muted, anchor="la")
    legend_x = 2190
    for label in resolved:
        color = palette[label]
        draw.line((legend_x, 104, legend_x + 68, 104), fill=color, width=8)
        draw.ellipse((legend_x + 27, 95, legend_x + 45, 113), fill=color, outline="white", width=2)
        draw.text((legend_x + 84, 104), label, font=fonts["legend"], fill=ink, anchor="lm")
        legend_x += 465

    panels = [
        (100, 225, 1080, 1030, "A", "Optimization", "Training cross-entropy", "Optimizer step"),
        (1160, 225, 2140, 1030, "B", "Generalization", "Validation cross-entropy", "Checkpoint step"),
        (2220, 225, 3200, 1030, "C", "Task performance", "BioASQ factoid MRR", "Checkpoint step"),
    ]
    boxes = []
    for x0, y0, x1, y1, letter, panel_title, y_label, x_label in panels:
        draw.rounded_rectangle((x0, y0, x1, y1), radius=16, fill=panel, outline=border, width=2)
        draw.ellipse((x0 + 14, y0 + 14, x0 + 68, y0 + 68), fill=ink)
        draw.text((x0 + 41, y0 + 41), letter, font=fonts["panel_label"], fill="white", anchor="mm")
        draw.text((x0 + 82, y0 + 41), panel_title, font=fonts["panel_title"], fill=ink, anchor="lm")
        box = (x0 + 120, y0 + 115, x1 - 38, y1 - 105)
        boxes.append(box)
        draw.line((box[0], box[3], box[2], box[3]), fill=ink, width=3)
        draw.line((box[0], box[1], box[0], box[3]), fill=ink, width=3)
        draw.text(((box[0] + box[2]) / 2, y1 - 48), x_label, font=fonts["axis"], fill=muted, anchor="mm")
        rotated = Image.new("RGBA", (600, 76), (0, 0, 0, 0))
        ImageDraw.Draw(rotated).text((300, 38), y_label, font=fonts["axis"], fill=muted, anchor="mm")
        rotated = rotated.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
        image.paste(rotated, (x0 - 60, int((box[1] + box[3]) / 2 - rotated.height / 2)), rotated)

    x_max = 225

    def scale_x(value, box):
        return box[0] + value / x_max * (box[2] - box[0])

    def scale_y(value, box, low, high, *, log=False):
        if log:
            value, low, high = math.log10(max(value, 1e-8)), math.log10(low), math.log10(high)
        return box[3] - (value - low) / (high - low) * (box[3] - box[1])

    def add_grid(box, y_ticks, low, high, formatter, *, log=False):
        for value in (0, 50, 100, 150, 200):
            x = scale_x(value, box)
            draw.line((x, box[1], x, box[3]), fill=grid_color, width=2)
            draw.text((x, box[3] + 18), str(value), font=fonts["tick"], fill=muted, anchor="ma")
        for value in y_ticks:
            y = scale_y(value, box, low, high, log=log)
            draw.line((box[0], y, box[2], y), fill=grid_color, width=2)
            draw.text((box[0] - 16, y), formatter(value), font=fonts["tick"], fill=muted, anchor="rm")

    add_grid(boxes[0], (.02, .05, .1, .2, .5, 1, 2), .015, 3.5, lambda x: f"{x:g}", log=True)
    add_grid(boxes[1], (.5, .6, .7, .8), .48, .82, lambda x: f"{x:.1f}")
    add_grid(boxes[2], (.325, .35, .375, .4, .425, .45, .475), .32, .49, lambda x: f"{x:.1%}")

    def line(points, color, line_width=5):
        if len(points) > 1:
            draw.line(points, fill=color, width=line_width, joint="curve")

    def dot(x, y, color, radius=8):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=3)

    def star(cx, cy, radius, color):
        points = []
        for index in range(10):
            angle = -math.pi / 2 + index * math.pi / 5
            current = radius if index % 2 == 0 else radius * .43
            points.append((cx + current * math.cos(angle), cy + current * math.sin(angle)))
        draw.polygon(points, fill=color, outline="white")

    def callout(x, y, text, color):
        text_width = draw.textbbox((0, 0), text, font=fonts["note"])[2]
        box = (x + 22, y - 53, x + text_width + 58, y - 5)
        draw.rounded_rectangle(box, radius=9, fill="white", outline=color, width=2)
        draw.text((x + 40, y - 29), text, font=fonts["note"], fill=color, anchor="lm")

    summaries = []
    for label, record in loaded.items():
        color = palette[label]
        metrics = record["metrics"]
        train = metrics.loc[metrics.loss.notna(), ["step", "loss"]]
        values = list(zip(train.step.astype(float), train.loss.astype(float)))
        raw = [(scale_x(step, boxes[0]), scale_y(loss, boxes[0], .015, 3.5, log=True)) for step, loss in values]
        line(raw, pale[label], 2)
        smooth = []
        for index, (step, _) in enumerate(values):
            start, stop = max(0, index - 5), min(len(values), index + 5)
            value = median(loss for _, loss in values[start:stop])
            smooth.append((scale_x(step, boxes[0]), scale_y(value, boxes[0], .015, 3.5, log=True)))
        line(smooth, color, 6)

        validation = metrics.loc[metrics.eval_loss.notna(), ["step", "eval_loss"]]
        validation_points = [
            (scale_x(float(row.step), boxes[1]), scale_y(float(row.eval_loss), boxes[1], .48, .82))
            for row in validation.itertuples()
        ]
        line(validation_points, color, 6)
        for x, y in validation_points:
            dot(x, y, color, 7)

        history_points = [
            (scale_x(float(row["step"]), boxes[2]), scale_y(float(row["metric_value"]), boxes[2], .32, .49))
            for row in record["history"]
        ]
        line(history_points, color, 6)
        for x, y in history_points:
            dot(x, y, color, 8)
        best = record["best"]
        best_x = scale_x(float(best["step"]), boxes[2])
        best_y = scale_y(float(best["metric_value"]), boxes[2], .32, .49)
        star(best_x, best_y, 22, color)
        callout(best_x, best_y, f'{best["metric_value"]:.1%} | step {best["step"]}', color)
        summaries.append({
            "model": label,
            "run": str(record["run"]),
            "best_step": int(best["step"]),
            "best_epoch": float(best["epoch"]),
            "best_dev_mrr": float(best["metric_value"]),
            "dev_questions": int(best["question_count"]),
        })

    draw.text((boxes[0][2] - 4, boxes[0][1] + 17), "10-step rolling median", font=fonts["note"], fill=muted, anchor="ra")
    draw.text((boxes[1][2] - 4, boxes[1][1] + 17), "Lower is better", font=fonts["note"], fill=muted, anchor="ra")
    draw.text((boxes[2][2] - 4, boxes[2][1] + 17), "Higher is better", font=fonts["note"], fill=muted, anchor="ra")
    draw.line((100, 1090, 3200, 1090), fill=border, width=2)
    caption = (
        "Training-time checkpoint selection | greedy decoding | "
        f"{max_seq_length:,}-token maximum sequence length | stars mark selected checkpoints"
    )
    draw.text((100, 1130), caption, font=fonts["note"], fill=muted, anchor="la")
    draw.text((3200, 1130), "Source: saved SFT training histories", font=fonts["note"], fill=muted, anchor="ra")

    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, optimize=True, dpi=(300, 300))
    return {"output_path": str(destination), "runs": summaries, "width": width, "height": height}
