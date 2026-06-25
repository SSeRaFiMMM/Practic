# -*- coding: utf-8 -*-
"""Индивидуальное задание SimpleAnalysis."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N_VALUES = 1000
LOW = -10_000
HIGH = 10_000
RANDOM_SEED = 42
BASE_DIR = Path(__file__).resolve().parent


class InfoBuffer:
    """Текстовый буфер для вывода Series.info()."""

    def __init__(self) -> None:
        self.chunks: list[str] = []

    def write(self, value: str) -> int:
        self.chunks.append(value)
        return len(value)


def generate_series(
    size: int = N_VALUES,
    low: int = LOW,
    high: int = HIGH,
    seed: int = RANDOM_SEED,
) -> pd.Series:
    """Сформировать Series из случайных целых чисел заданного диапазона."""
    rng = np.random.default_rng(seed)
    values = rng.integers(low=low, high=high + 1, size=size)
    return pd.Series(values, name="values")


def save_dataset(series: pd.Series, output_dir: Path) -> tuple[Path, Path]:
    """Сохранить исходный набор данных в файлах CSV и TXT."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "dataset.csv"
    txt_path = output_dir / "dataset.txt"
    series.to_csv(csv_path, index_label="index", encoding="utf-8")
    np.savetxt(txt_path, series.to_numpy(), fmt="%d")
    return csv_path, txt_path


def get_summary(series: pd.Series) -> str:
    """Вернуть результаты предварительного анализа Series."""
    buffer = InfoBuffer()
    series.info(buf=buffer)
    return (
        "Первые 10 элементов набора данных:\n"
        f"{series.head(10).to_string()}\n\n"
        "Сводная информация о структуре Series:\n"
        f"{''.join(buffer.chunks)}\n"
        "Описательная статистика:\n"
        f"{series.describe().to_string()}"
    )


def clean_series(raw: pd.Series, low: int = LOW, high: int = HIGH) -> pd.Series:
    """Очистить данные от пропусков, нечисловых значений и выбросов."""
    numeric = pd.to_numeric(raw, errors="coerce")
    without_gaps = numeric.dropna()
    in_range = without_gaps[(without_gaps >= low) & (without_gaps <= high)]
    return in_range.astype(int).reset_index(drop=True)


def make_contaminated_series(series: pd.Series) -> pd.Series:
    """Создать демонстрационную копию с цифровым мусором."""
    garbage = pd.Series([np.nan, "abc", "12 345", 999_999, -50_000, None], dtype=object)
    return pd.concat([series.astype(object), garbage], ignore_index=True)


def count_repeated_values(series: pd.Series) -> tuple[int, int]:
    """Вернуть число повторяющихся уникальных значений и лишних повторов."""
    counts = series.value_counts()
    repeated = counts[counts > 1]
    return int(repeated.size), int((repeated - 1).sum())


def compute_statistics(series: pd.Series) -> dict[str, Any]:
    """Рассчитать требуемые характеристики набора."""
    repeated_values, extra_repeats = count_repeated_values(series)
    return {
        "minimum": int(series.min()),
        "repeated_values": repeated_values,
        "extra_repeats": extra_repeats,
        "maximum": int(series.max()),
        "total_sum": int(series.sum()),
        "std_dev": float(np.std(series.to_numpy(), ddof=0)),
        "mean": float(series.mean()),
    }


def format_statistics(statistics: dict[str, Any]) -> str:
    """Подготовить пояснительный текст с числовыми характеристиками."""
    return (
        "СТАНДАРТНЫЕ ЧИСЛОВЫЕ ХАРАКТЕРИСТИКИ НАБОРА ДАННЫХ\n"
        "============================================================\n"
        f"1. Минимальное значение                : {statistics['minimum']}\n"
        f"2. Количество повторяющихся значений   : {statistics['repeated_values']}\n"
        f"   Избыточных повторов                 : {statistics['extra_repeats']}\n"
        f"3. Максимальное значение               : {statistics['maximum']}\n"
        f"4. Сумма всех чисел                    : {statistics['total_sum']}\n"
        f"5. Среднеквадратическое отклонение     : {statistics['std_dev']:.2f}\n"
        f"   Среднее арифметическое              : {statistics['mean']:.2f}"
    )


def round_to_hundreds(value: float) -> int:
    """Округлить до сотен по математическому правилу, от нуля при 0,5."""
    sign = 1 if value >= 0 else -1
    return sign * int(math.floor(abs(value) / 100 + 0.5)) * 100


def round_series_to_hundreds(series: pd.Series) -> pd.Series:
    """Применить математическое округление до сотен к каждому значению."""
    return series.map(round_to_hundreds)


def build_dataframe(series: pd.Series) -> pd.DataFrame:
    """Сформировать DataFrame с исходным и отсортированными рядами."""
    frame = pd.DataFrame({"values": series})
    frame["sorted_asc"] = series.sort_values().reset_index(drop=True)
    frame["sorted_desc"] = series.sort_values(ascending=False).reset_index(drop=True)
    return frame


def finish_plot(fig: plt.Figure, path: Path, show_plots: bool) -> Path:
    """Сохранить график, при необходимости показать его и освободить ресурсы."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    if show_plots and "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)
    return path


def plot_line(series: pd.Series, figures_dir: Path, show_plots: bool = True) -> Path:
    """Построить линейный график исходных данных."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(series.index, series.values, linewidth=0.8)
    ax.set_title("Линейный график исходного набора данных")
    ax.set_xlabel("Индекс элемента")
    ax.set_ylabel("Значение")
    ax.grid(True)
    return finish_plot(fig, figures_dir / "01_line_plot.png", show_plots)


def plot_histogram(series: pd.Series, figures_dir: Path, show_plots: bool = True) -> Path:
    """Построить прямоугольную гистограмму округленных значений."""
    rounded = round_series_to_hundreds(series)
    bins = np.arange(LOW - 50, HIGH + 151, 100)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(rounded, bins=bins, edgecolor="black")
    ax.set_title("Гистограмма распределения значений\n(округление до сотен)")
    ax.set_xlabel("Значение, округленное до сотен")
    ax.set_ylabel("Частота")
    ax.grid(axis="y")
    return finish_plot(fig, figures_dir / "02_histogram.png", show_plots)


def plot_sorted(frame: pd.DataFrame, figures_dir: Path, show_plots: bool = True) -> Path:
    """Построить совместный график рядов, отсортированных в двух направлениях."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(frame.index, frame["sorted_asc"], label="По возрастанию", linewidth=1.2)
    ax.plot(frame.index, frame["sorted_desc"], label="По убыванию", linewidth=1.2)
    ax.set_title("Отсортированные значения набора данных")
    ax.set_xlabel("Порядковый номер")
    ax.set_ylabel("Значение")
    ax.grid(True)
    ax.legend()
    return finish_plot(fig, figures_dir / "03_sorted_plot.png", show_plots)


def run_analysis(output_dir: Path = BASE_DIR, show_plots: bool = True) -> dict[str, Any]:
    """Выполнить полный сценарий и сохранить все результаты."""
    figures_dir = output_dir / "figures"
    series = generate_series()
    csv_path, txt_path = save_dataset(series, output_dir)
    contaminated = make_contaminated_series(series)
    cleaned = clean_series(contaminated)
    statistics = compute_statistics(cleaned)
    frame = build_dataframe(cleaned)
    plots = [
        plot_line(cleaned, figures_dir, show_plots),
        plot_histogram(cleaned, figures_dir, show_plots),
        plot_sorted(frame, figures_dir, show_plots),
    ]
    results_text = (
        "ИНДИВИДУАЛЬНОЕ ЗАДАНИЕ SimpleAnalysis\n\n"
        "[1] Получение набора данных\n"
        f"Сформирован объект Series: {len(series)} целых чисел в диапазоне [{LOW}; {HIGH}].\n"
        f"Данные сохранены: {csv_path.name}, {txt_path.name}.\n\n"
        f"{get_summary(series)}\n\n"
        "[2] Очистка данных от цифрового мусора\n"
        f"Размер загрязненного набора: {len(contaminated)}.\n"
        f"Размер очищенного набора: {len(cleaned)}.\n"
        f"Удалено элементов: {len(contaminated) - len(cleaned)}.\n\n"
        "[3] Расчет числовых характеристик\n"
        f"{format_statistics(statistics)}\n\n"
        "[4] Формирование DataFrame\n"
        f"{frame.head(10).to_string()}\n\n"
        "[5] Визуализация\n"
        + "\n".join(f"Сформирован файл: {path.relative_to(output_dir)}" for path in plots)
        + "\n"
    )
    results_path = output_dir / "results.txt"
    results_path.write_text(results_text, encoding="utf-8")
    return {
        "series": series,
        "cleaned": cleaned,
        "statistics": statistics,
        "dataframe": frame,
        "csv_path": csv_path,
        "txt_path": txt_path,
        "results_path": results_path,
        "plots": plots,
    }


def main() -> None:
    result = run_analysis()
    print(result["results_path"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
