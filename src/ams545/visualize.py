import math
from typing import List, Tuple

import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from graph.segment import Segment
from graph.vector import Vector
from graph.trapezoid import Trapezoid
from src.ams545.trapmap import TrapMap
from src.ams545.utils import _SHEAR

def _unshear(x: float, y: float) -> Tuple[float, float]:
    """Convert a point from sheared space back to original coordinates."""
    return x - _SHEAR, y


def _trap_verts_original(trap: Trapezoid) -> List[Tuple[float, float]]:
    """*trap* in original coordinates."""
    return [_unshear(v.x, v.y) for v in trap.get_boundary_vertices()]


def _seg_original(seg: Segment) -> Tuple[float, float, float, float]:
    """*seg* in original coordinates."""
    lx, ly = _unshear(seg.left_point.x, seg.left_point.y)
    rx, ry = _unshear(seg.right_point.x, seg.right_point.y)
    return lx, ly, rx, ry

class TrapMapVisualizer:
    _TRAP_ALPHA = 0.38
    _TRAP_EDGE = "#444444"
    _SEG_COLOR = "#1a1a2e"
    _SEG_WIDTH = 1.8
    _HL_COLOR  = "#e63946"  
    _HL_WIDTH  = 2.8
    _DOT_SIZE  = 4

    def __init__(
        self,
        segments: List[Segment],
        title: str = "Trapezoidal Map Construction",
        shuffle: bool = False,
    ) -> None:
        self._title = title

        # Snapshots: one entry per segment insertion.
        # Each entry is (trapezoids_snapshot, sheared_segment_just_added).
        self._snapshots: List[Tuple[List[Trapezoid], Segment]] = []
        self._current_step: int = 0

        def _callback(trapmap: TrapMap, _step: int, seg: Segment) -> None:
            self._snapshots.append((list(trapmap.get_all_trapezoids()), seg))

        if isinstance(segments[0], Segment):  # type: ignore[comparison-overlap]
            self._trapmap = TrapMap(segments, step_callback=_callback, shuffle=shuffle)
        else:
            self._trapmap = TrapMap.from_polygons(segments, step_callback=_callback, shuffle=shuffle)
        self._xlim, self._ylim = _axis_limits(segments)

        if not self._snapshots:
            raise ValueError("No segments were inserted")

        self._root = tk.Tk()
        self._root.title(title)
        self._root.minsize(700, 520)
        self._setup_ui()
        self._draw_step(0)

    def _setup_ui(self) -> None:
        self._fig, self._ax = plt.subplots(figsize=(10, 7))
        self._fig.patch.set_facecolor("#f8f8f8")

        # Main content area: canvas left, info panel right
        content = ttk.Frame(self._root)
        content.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self._canvas = FigureCanvasTkAgg(self._fig, master=content)
        self._canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        side = ttk.LabelFrame(content, text="Info", padding=8, width=210)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 0))
        side.pack_propagate(False)
        
        cords = ttk.Frame(side)
        cords.pack(fill=tk.X, pady=(0, 8))
        self._gps_coords_question = ttk.Label(cords, text="Input GPS coords: ", font=("Arial", 10))
        self._gps_coords_question.pack(anchor=tk.NW, pady=(0, 8))
        
        self._input_gps_coords = ttk.Entry(cords, font=("Arial", 10, "bold"))
        self._input_gps_coords.insert(tk.END, "45.123, -93.456")  # Placeholder text
        self._input_gps_coords.pack(fill=tk.X, pady=(0, 8))

        region = ttk.Frame(side)
        region.pack(fill=tk.X, pady=(0, 8))
        self._containing_region = ttk.Label(region, text="Containing region: ", font=("Arial", 10))
        self._containing_region.pack(anchor=tk.NW, pady=(0, 8))

        self._containing_region_answer = ttk.Label(region, text="", wraplength=200, font=("Arial", 10, "bold"))
        self._containing_region_answer.pack(anchor=tk.NW, pady=(0, 8))
        
        # --- Navigation bar ---
        bar = ttk.Frame(self._root)
        bar.pack(fill=tk.X, padx=10, pady=(0, 10))

        self._btn_prev = ttk.Button(bar, text="Previous", command=self._go_prev)
        self._btn_prev.pack(side=tk.LEFT, padx=4)

        self._btn_next = ttk.Button(bar, text="Next", command=self._go_next)
        self._btn_next.pack(side=tk.RIGHT, padx=4)

        self._btn_quit = ttk.Button(bar, text="Quit", command=self.quit)
        self._btn_quit.pack(side=tk.RIGHT, padx=4)

        # Keyboard shortcuts
        self._root.bind("<Left>",  lambda _e: self._go_prev())
        self._root.bind("<Right>", lambda _e: self._go_next())
        self._root.bind("<Escape>", lambda _e: self.quit())
        self._root.bind("<Return>", lambda _e: self.get_cordinates())

    def _go_prev(self) -> None:
        if self._current_step > 0:
            self._current_step -= 1
            self._draw_step(self._current_step)

    def _go_next(self) -> None:
        if self._current_step < len(self._snapshots) - 1:
            self._current_step += 1
            self._draw_step(self._current_step)

    def _draw_step(self, step: int) -> None:
        ax = self._ax
        ax.cla()
        ax.set_facecolor("#fcfcfc")

        traps, cur_seg = self._snapshots[step]
        n_traps = len(traps)

        # Colour map for trapezoids
        cmap = plt.cm.tab20 if n_traps > 12 else plt.cm.Set3
        colours = cmap(np.linspace(0, 1, max(n_traps, 1)))

        # Draw trapezoids
        for i, trap in enumerate(traps):
            try:
                verts = _trap_verts_original(trap)
                patch = plt.Polygon(
                    verts, closed=True,
                    facecolor=colours[i % len(colours)],
                    alpha=self._TRAP_ALPHA,
                    edgecolor=self._TRAP_EDGE,
                    linewidth=0.9,
                    zorder=1,
                )
                ax.add_patch(patch)
            except Exception:
                pass

        for prev_step in range(step):
            seg = self._snapshots[prev_step][1]
            lx, ly, rx, ry = _seg_original(seg)
            ax.plot(
                [lx, rx], [ly, ry],
                color=self._SEG_COLOR, linewidth=self._SEG_WIDTH,
                solid_capstyle="round", zorder=3,
            )
            ax.plot(
                [lx, lx, rx, rx], [ly, ly, ry, ry],
                linestyle="none", marker="o",
                markersize=self._DOT_SIZE, color=self._SEG_COLOR, zorder=3,
            )

        lx,ly,rx,ry = _seg_original(cur_seg)
        ax.plot(
            [lx, rx], [ly, ry],
            color=self._HL_COLOR, linewidth=self._HL_WIDTH,
            solid_capstyle="round", zorder=4,
        )
        ax.plot(
            [lx, lx, rx, rx], [ly, ly, ry, ry],
            linestyle="none", marker="o",
            markersize=self._DOT_SIZE + 1, color=self._HL_COLOR, zorder=4,
        )

        # Reset axes
        ax.set_xlim(*self._xlim)
        ax.set_ylim(*self._ylim)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x", fontsize=10)
        ax.set_ylabel("y", fontsize=10)
        ax.set_title(
            f"Step {step + 1} / {len(self._snapshots)}  —  "
            f"added ({lx:.3g}, {ly:.3g}) → ({rx:.3g}, {ry:.3g})",
            fontsize=11,
        )
        ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.55, zorder=0)

        self._fig.tight_layout()
        self._canvas.draw()

        n = len(self._snapshots)
        self._btn_prev.state(["disabled"] if step == 0       else ["!disabled"])
        self._btn_next.state(["disabled"] if step == n - 1  else ["!disabled"])

    def get_cordinates(self) -> Tuple[str, str]:
        input_coords = self._input_gps_coords.get()
        x, y = input_coords.split(",")
        x, y = float(x.strip()), float(y.strip())
        result = self._trapmap.find_containing_polygon(x, y)
        self._containing_region_answer.config(text=str(result) if result is not None else "Not found")
        return input_coords, result

    def run(self) -> None:
        """Start the Tkinter event loop """
        self._root.mainloop()
    
    def quit(self) -> None:
        self._root.quit()
        self._root.destroy()

def _axis_limits(
    segments, pad_frac: float = 0.15
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if not segments:
        return (-1.0, 1.0), (-1.0, 1.0)

    if isinstance(segments[0], Segment):
        xs = [s.left_point.x for s in segments] + [s.right_point.x for s in segments]
        ys = [s.left_point.y for s in segments] + [s.right_point.y for s in segments]
    else:
        # List of polygons (each polygon is a List[Vector])
        xs = [v.x for poly in segments for v in poly]
        ys = [v.y for poly in segments for v in poly]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    dx = max(xmax - xmin, 1e-3)
    dy = max(ymax - ymin, 1e-3)
    px, py = dx * pad_frac, dy * pad_frac

    return (xmin - px, xmax + px), (ymin - py, ymax + py)


def visualize_trapmap_construction(
    segments: List[Segment],
    title: str = "Trapezoidal Map Construction",
    shuffle: bool = True,
) -> None:
    TrapMapVisualizer(segments, title=title, shuffle=shuffle).run()


if __name__ == "__main__":
    # demo_segs = [
    #     # Degeneracy testing
    #     Segment.from_coords(0,0,100,0),
    #     Segment.from_coords(0,100,100,100),
    #     Segment.from_coords(0,0,0,100),
    #     Segment.from_coords(100,0,100,100),
    #     Segment.from_coords(25,25,75,25),
    #     Segment.from_coords(25,25,25,75),
    #     Segment.from_coords(75,25,75,75),
    #     Segment.from_coords(25,75,75,75),
    # ]
    demo_segs = [Vector(0,0), Vector(50,0), Vector(50,50), Vector(0,50)]   # left square

    visualize_trapmap_construction([demo_segs], title="Trapezoidal Map — Demo", shuffle=True)
