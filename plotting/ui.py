from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from matplotlib.widgets import Button, TextBox

Rect = tuple[float, float, float, float]


@dataclass
class VBox:
    x: float
    y: float
    w: float
    h: float
    gap: float = 0.02
    padding: float = 0.0

    def __post_init__(self) -> None:
        self.x = self.x + self.padding
        self.y = self.y + self.padding
        self.w = max(0.0, self.w - 2 * self.padding)
        self.h = max(0.0, self.h - 2 * self.padding)
        self._cursor = self.y + self.h

    def next(self, height: float) -> Rect:
        height = max(0.0, min(height, self._cursor - self.y))
        y = self._cursor - height
        rect = (self.x, y, self.w, height)
        self._cursor = y - self.gap
        return rect

    def next_frac(self, fraction: float) -> Rect:
        return self.next(self.h * float(fraction))


@dataclass
class HBox:
    x: float
    y: float
    w: float
    h: float
    gap: float = 0.02
    padding: float = 0.0

    def __post_init__(self) -> None:
        self.x = self.x + self.padding
        self.y = self.y + self.padding
        self.w = max(0.0, self.w - 2 * self.padding)
        self.h = max(0.0, self.h - 2 * self.padding)
        self._cursor = self.x

    def next(self, width: float) -> Rect:
        width = max(0.0, min(width, self.x + self.w - self._cursor))
        rect = (self._cursor, self.y, width, self.h)
        self._cursor = self._cursor + width + self.gap
        return rect

    def next_frac(self, fraction: float) -> Rect:
        return self.next(self.w * float(fraction))


def split_vertical(rect: Rect, top_frac: float) -> tuple[Rect, Rect]:
    x, y, w, h = rect
    top_h = max(0.0, min(h, h * float(top_frac)))
    top_rect = (x, y + h - top_h, w, top_h)
    bottom_rect = (x, y, w, h - top_h)
    return top_rect, bottom_rect


def split_horizontal(rect: Rect, left_frac: float) -> tuple[Rect, Rect]:
    x, y, w, h = rect
    left_w = max(0.0, min(w, w * float(left_frac)))
    left_rect = (x, y, left_w, h)
    right_rect = (x + left_w, y, w - left_w, h)
    return left_rect, right_rect


def style_panel(ax, *, facecolor: str = "0.96", axis_off: bool = True) -> None:
    if axis_off:
        ax.set_axis_off()
    ax.set_facecolor(facecolor)


def add_text(ax, x: float, y: float, text: str, **kwargs):
    if "transform" not in kwargs:
        kwargs["transform"] = ax.transAxes
    return ax.text(x, y, text, **kwargs)


def add_button(ax, rect: Rect, label: str, **kwargs) -> Button:
    button_ax = ax.inset_axes(rect)
    return Button(button_ax, label, **kwargs)


def add_textbox(
    ax,
    rect: Rect,
    *,
    label: str = "",
    initial: str = "",
    hide_label: bool = True,
    **kwargs,
) -> TextBox:
    textbox_ax = ax.inset_axes(rect)
    textbox = TextBox(textbox_ax, label, initial=initial, **kwargs)
    if hide_label:
        textbox.label.set_visible(False)
    return textbox


@dataclass
class Dropdown:
    fig: object
    parent_ax: object
    button_rect: Rect
    list_rect: Rect
    items: Sequence[str]
    on_select: Callable[[str], None]
    label: str = "v"
    font_size: float = 10.0
    padding: float = 0.01
    facecolor: str = "0.98"
    text_color: str = "0.1"

    def __post_init__(self) -> None:
        self.button = add_button(self.parent_ax, self.button_rect, self.label)
        self.list_ax = self.parent_ax.inset_axes(self.list_rect)
        style_panel(self.list_ax, facecolor=self.facecolor, axis_off=True)
        self.list_ax.set_visible(False)
        self._text_items: list = []
        self.button.on_clicked(lambda _event: self.toggle())
        self._pick_cid = self.fig.canvas.mpl_connect("pick_event", self._on_pick)

    def set_items(self, items: Sequence[str]) -> None:
        self.items = list(items)
        if self.list_ax.get_visible():
            self._build_list()
            self.fig.canvas.draw_idle()

    def show(self) -> None:
        self._build_list()
        self.list_ax.set_visible(True)
        self.fig.canvas.draw_idle()

    def hide(self) -> None:
        self.list_ax.set_visible(False)
        self.fig.canvas.draw_idle()

    def toggle(self) -> None:
        if self.list_ax.get_visible():
            self.hide()
        else:
            self.show()

    def _build_list(self) -> None:
        labels = list(self.items) if self.items else ["(none)"]
        self.list_ax.clear()
        style_panel(self.list_ax, facecolor=self.facecolor, axis_off=True)
        self.list_ax.set_xlim(0.0, 1.0)
        self.list_ax.set_ylim(0.0, 1.0)
        self._text_items = []
        for label in labels:
            text = self.list_ax.text(
                0.02,
                0.0,
                label,
                transform=self.list_ax.transAxes,
                ha="left",
                va="top",
                fontsize=self.font_size,
                color=self.text_color,
                picker=True,
            )
            self._text_items.append(text)
        self.list_ax.set_visible(True)
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        inv = self.list_ax.transAxes.inverted()
        heights = []
        for text in self._text_items:
            bbox = text.get_window_extent(renderer=renderer)
            bbox_ax = bbox.transformed(inv)
            heights.append(float(bbox_ax.height))
        y_pos = 0.96
        for text, height in zip(self._text_items, heights):
            text.set_position((0.02, y_pos))
            y_pos -= height + float(self.padding)

    def _on_pick(self, event) -> None:
        if not self.list_ax.get_visible():
            return
        artist = event.artist
        if artist not in self._text_items:
            return
        label = artist.get_text()
        self.on_select(label)
        self.hide()
