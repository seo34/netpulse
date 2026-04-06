#!/usr/bin/env python3
"""
NetPulse — HTTP Proxy Performance Tester
WinMTR-style continuous HTTP/HTTPS proxy benchmarking tool.

Usage:
    python main.py

Requirements:
    pip install PyQt5 aiohttp matplotlib numpy
"""

import sys
import asyncio
import aiohttp
import time
import csv
import math
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import deque
from datetime import datetime
from urllib.parse import urlparse

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QPushButton, QLineEdit, QLabel,
    QDoubleSpinBox, QSpinBox, QSplitter, QGroupBox,
    QHeaderView, QFileDialog, QMessageBox, QListWidget, QListWidgetItem,
    QAbstractItemView, QStatusBar, QFrame, QSizePolicy, QToolTip,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QColor, QFont, QBrush, QPalette

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_URL         = "https://www.walmart.com"
DEFAULT_INTERVAL    = 1.0
DEFAULT_TIMEOUT     = 15.0
DEFAULT_CONCURRENCY = 1
MAX_HISTORY         = 120

PROXY_COLORS = [
    "#2196F3", "#4CAF50", "#FF5722", "#9C27B0",
    "#FF9800", "#00BCD4", "#E91E63", "#8BC34A",
    "#795548", "#607D8B", "#F44336", "#3F51B5",
]

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

TABLE_COLS = [
    "Proxy", "Avg (ms)", "Min (ms)", "Max (ms)",
    "Jitter (ms)", "Requests", "Failures", "Loss %", "Last (ms)", "Status",
]


# ── Data Model ────────────────────────────────────────────────────────────────

@dataclass
class ProxyStats:
    proxy_id:  str
    raw:       str          # original input string
    proxy_url: str          # http://ip:port  (no credentials)
    proxy_auth: object      # aiohttp.BasicAuth or None
    color:     str          # hex colour for graph

    total_requests: int = 0
    failures:       int = 0
    last_error:     str = ""

    _times: list = field(default_factory=list)          # successful total times (seconds)

    # Rolling history for graph (ms; None = failure)
    history_total: deque = field(default_factory=lambda: deque(maxlen=MAX_HISTORY))
    history_ts:    deque = field(default_factory=lambda: deque(maxlen=MAX_HISTORY))

    def record(self, total: Optional[float], error: str = ""):
        self.total_requests += 1
        ts = time.time()
        if total is None:
            self.failures += 1
            self.last_error = error or "Error"
            self.history_total.append(None)
        else:
            self.last_error = ""
            self._times.append(total)
            if len(self._times) > 1000:
                self._times = self._times[-500:]
            self.history_total.append(total * 1000)
        self.history_ts.append(ts)

    # ── computed stats ─────────────────────────────────────────────────────

    @property
    def avg_ms(self) -> float:
        return (sum(self._times) / len(self._times) * 1000) if self._times else 0.0

    @property
    def min_ms(self) -> float:
        return min(self._times) * 1000 if self._times else 0.0

    @property
    def max_ms(self) -> float:
        return max(self._times) * 1000 if self._times else 0.0

    @property
    def jitter_ms(self) -> float:
        return float(np.std(self._times)) * 1000 if len(self._times) >= 2 else 0.0

    @property
    def loss_pct(self) -> float:
        return (self.failures / self.total_requests * 100) if self.total_requests else 0.0

    @property
    def display_name(self) -> str:
        try:
            p = urlparse(self.proxy_url)
            return f"{p.hostname}:{p.port}"
        except Exception:
            return self.proxy_url

    @property
    def last_ms(self) -> Optional[float]:
        return self.history_total[-1] if self.history_total else None


# ── Proxy Parsing ─────────────────────────────────────────────────────────────

def parse_proxy(raw: str) -> Tuple[str, object]:
    """
    Accept any of:
      ip:port
      ip:port:user:pass
      user:pass@ip:port
      http://ip:port
      http://user:pass@ip:port
    Returns (proxy_url_no_auth, aiohttp.BasicAuth_or_None).
    """
    s = raw.strip()
    if not s:
        raise ValueError("Empty proxy string")

    # Handle ip:port:user:pass format
    if "://" not in s and "@" not in s:
        parts = s.split(":")
        if len(parts) == 4:
            # ip:port:user:pass
            host, port_str, user, password = parts
            try:
                port = int(port_str)
                return f"http://{host}:{port}", aiohttp.BasicAuth(login=user, password=password)
            except ValueError:
                pass

    if "://" not in s:
        s = "http://" + s

    parsed = urlparse(s)
    host = parsed.hostname
    port = parsed.port or 8080
    proxy_url = f"http://{host}:{port}"

    auth = None
    if parsed.username:
        auth = aiohttp.BasicAuth(
            login=parsed.username,
            password=parsed.password or "",
        )
    return proxy_url, auth


# ── Worker Thread ─────────────────────────────────────────────────────────────

class TestWorker(QThread):
    """
    Background QThread that owns an asyncio event loop.
    Emits result_ready(proxy_id, result_dict) after every individual request.
    """
    result_ready = pyqtSignal(str, dict)

    def __init__(
        self,
        proxy_list: List[ProxyStats],
        target_url: str,
        interval: float,
        timeout: float,
        concurrency: int,
    ):
        super().__init__()
        self.proxy_list  = proxy_list
        self.target_url  = target_url
        self.interval    = interval
        self.timeout     = timeout
        self.concurrency = concurrency
        self._running    = True
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def stop(self):
        self._running = False
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)

    def run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main_loop())
        except Exception:
            pass
        finally:
            try:
                self._loop.close()
            except Exception:
                pass

    async def _main_loop(self):
        while self._running:
            cycle_start = time.time()
            tasks = [
                self._test_one(ps)
                for ps in self.proxy_list
                for _ in range(self.concurrency)
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = time.time() - cycle_start
            sleep_t = max(0.0, self.interval - elapsed)
            if sleep_t > 0 and self._running:
                await asyncio.sleep(sleep_t)

    async def _test_one(self, ps: ProxyStats):
        if not self._running:
            return

        req_start = time.time()
        first_byte: List[Optional[float]] = [None]
        first_byte_seen = [False]

        async def on_chunk(session, ctx, params):
            if not first_byte_seen[0]:
                first_byte_seen[0] = True
                first_byte[0] = time.time()

        trace = aiohttp.TraceConfig()
        trace.on_response_chunk_received.append(on_chunk)

        timeout_cfg = aiohttp.ClientTimeout(
            total=self.timeout,
            connect=min(self.timeout, 10.0),
        )

        try:
            conn = aiohttp.TCPConnector(ssl=False, force_close=True, limit=0)
            async with aiohttp.ClientSession(
                trace_configs=[trace],
                connector=conn,
                timeout=timeout_cfg,
            ) as session:
                kw: dict = {
                    "proxy":   ps.proxy_url,
                    "headers": REQUEST_HEADERS,
                    "allow_redirects": True,
                }
                if ps.proxy_auth:
                    kw["proxy_auth"] = ps.proxy_auth

                async with session.get(self.target_url, **kw) as resp:
                    await resp.read()
                    total = time.time() - req_start
                    ttfb  = (first_byte[0] - req_start) if first_byte[0] else total
                    self.result_ready.emit(ps.proxy_id, {
                        "success": True,
                        "total":   total,
                        "ttfb":    ttfb,
                        "status":  resp.status,
                    })

        except asyncio.TimeoutError:
            self.result_ready.emit(ps.proxy_id, {
                "success": False,
                "error":   "Timeout",
                "elapsed": time.time() - req_start,
            })
        except aiohttp.ClientProxyConnectionError as exc:
            self.result_ready.emit(ps.proxy_id, {
                "success": False,
                "error":   f"ProxyConn: {str(exc)[:50]}",
            })
        except aiohttp.ClientConnectorError as exc:
            self.result_ready.emit(ps.proxy_id, {
                "success": False,
                "error":   f"ConnErr: {str(exc)[:50]}",
            })
        except Exception as exc:
            self.result_ready.emit(ps.proxy_id, {
                "success": False,
                "error":   str(exc)[:60],
            })


# ── Graph Widget ──────────────────────────────────────────────────────────────

class GraphWidget(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(9, 3.2), dpi=100, facecolor="#1a1a2e")
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.15)
        self._proxy_list: List[ProxyStats] = []
        self._draw_empty()

    def _draw_empty(self):
        ax = self.ax
        ax.set_facecolor("#16213e")
        ax.set_title("Real-time Response Times", color="#e0e0e0", fontsize=10, pad=8)
        ax.set_xlabel("Request #", color="#aaaaaa", fontsize=8)
        ax.set_ylabel("ms", color="#aaaaaa", fontsize=8)
        ax.tick_params(colors="#888888", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#333355")
        ax.grid(True, alpha=0.2, color="#444466", linestyle="--")
        self.draw()

    def set_proxies(self, proxy_list: List[ProxyStats]):
        self._proxy_list = proxy_list

    def refresh(self):
        ax = self.ax
        ax.clear()
        ax.set_facecolor("#16213e")
        ax.set_title("Real-time Response Times", color="#e0e0e0", fontsize=10, pad=8)
        ax.set_xlabel("Request #", color="#aaaaaa", fontsize=8)
        ax.set_ylabel("ms", color="#aaaaaa", fontsize=8)
        ax.tick_params(colors="#888888", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#333355")
        ax.grid(True, alpha=0.2, color="#444466", linestyle="--")

        any_data = False

        for ps in self._proxy_list:
            history = list(ps.history_total)
            if not history:
                continue

            x_all  = list(range(len(history)))
            x_good = [i for i, v in enumerate(history) if v is not None]
            y_good = [v for v in history if v is not None]

            if not y_good:
                continue

            any_data = True

            # main line
            ax.plot(
                x_good, y_good,
                color=ps.color, linewidth=1.6, alpha=0.9,
                label=ps.display_name,
                marker="o", markersize=2.5, markerfacecolor=ps.color,
            )

            # failure markers (red X on x-axis)
            x_fail = [i for i, v in enumerate(history) if v is None]
            if x_fail:
                y_fail = [0] * len(x_fail)
                ax.scatter(
                    x_fail, y_fail,
                    marker="x", color="#ff4444", s=40, zorder=5,
                    linewidths=1.5,
                )

            # spike markers (red triangle up)
            if len(y_good) > 3:
                avg = sum(y_good) / len(y_good)
                sx = [xi for xi, yi in zip(x_good, y_good) if yi > 2 * avg]
                sy = [yi for xi, yi in zip(x_good, y_good) if yi > 2 * avg]
                if sx:
                    ax.scatter(
                        sx, sy,
                        marker="^", color="#FF6B6B", s=55, zorder=6, alpha=0.85,
                    )

        if any_data:
            legend = ax.legend(
                fontsize=7.5, loc="upper left",
                facecolor="#1a1a2e", edgecolor="#444466",
                labelcolor="#dddddd",
            )

        self.fig.canvas.draw_idle()


# ── Stats Table ───────────────────────────────────────────────────────────────

class StatsTable(QTableWidget):
    def __init__(self):
        super().__init__()
        self.setColumnCount(len(TABLE_COLS))
        self.setHorizontalHeaderLabels(TABLE_COLS)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setAlternatingRowColors(False)
        self.setShowGrid(True)
        self.verticalHeader().setVisible(False)
        self.setFont(QFont("Consolas", 9))
        self.setMinimumHeight(160)

        hdr = self.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, len(TABLE_COLS)):
            hdr.setSectionResizeMode(i, QHeaderView.ResizeToContents)

        self._apply_style()

    def _apply_style(self):
        self.setStyleSheet("""
            QTableWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
                gridline-color: #313244;
                border: 1px solid #313244;
            }
            QTableWidget::item {
                padding: 2px 6px;
            }
            QTableWidget::item:selected {
                background-color: #45475a;
                color: #cdd6f4;
            }
            QHeaderView::section {
                background-color: #181825;
                color: #89b4fa;
                padding: 4px 6px;
                border: none;
                border-bottom: 1px solid #45475a;
                font-weight: bold;
                font-size: 9px;
            }
        """)

    def _cell(self, text: str, align=Qt.AlignCenter, fg=None, bg=None, bold=False) -> QTableWidgetItem:
        item = QTableWidgetItem(str(text))
        item.setTextAlignment(align | Qt.AlignVCenter)
        if fg:
            item.setForeground(QBrush(QColor(fg)))
        if bg:
            item.setBackground(QBrush(QColor(bg)))
        if bold:
            f = QFont("Consolas", 9, QFont.Bold)
            item.setFont(f)
        return item

    def rebuild(self, proxy_list: List[ProxyStats]):
        self.setRowCount(len(proxy_list))
        for row, ps in enumerate(proxy_list):
            self._fill_row(row, ps)

    def update_all(self, proxy_list: List[ProxyStats]):
        for row, ps in enumerate(proxy_list):
            if row < self.rowCount():
                self._fill_row(row, ps)

    def _fill_row(self, row: int, ps: ProxyStats):
        # column 0 — proxy name (coloured)
        name_item = self._cell(ps.display_name, Qt.AlignLeft, fg=ps.color, bold=True)
        name_item.setToolTip(ps.raw)
        self.setItem(row, 0, name_item)

        avg = ps.avg_ms
        last = ps.last_ms

        # detect spike
        is_spike = (
            last is not None and avg > 0 and last > 2 * avg
        )

        def fmt(v: float) -> str:
            return f"{v:.1f}" if v else "—"

        self.setItem(row, 1, self._cell(fmt(avg)))
        self.setItem(row, 2, self._cell(fmt(ps.min_ms) if ps._times else "—"))
        self.setItem(row, 3, self._cell(fmt(ps.max_ms) if ps._times else "—"))
        self.setItem(row, 4, self._cell(fmt(ps.jitter_ms) if len(ps._times) >= 2 else "—"))
        self.setItem(row, 5, self._cell(str(ps.total_requests)))

        # failures
        fail_bg = "#3d1a1a" if ps.failures > 0 else None
        self.setItem(row, 6, self._cell(str(ps.failures), bg=fail_bg))

        # loss %
        lp = ps.loss_pct
        loss_bg = "#3d1a1a" if lp > 10 else ("#3d3010" if lp > 0 else None)
        self.setItem(row, 7, self._cell(f"{lp:.1f}%", bg=loss_bg))

        # last
        if last is None:
            self.setItem(row, 8, self._cell("FAIL", bg="#3d1a1a", fg="#ff6b6b"))
        else:
            spike_bg = "#3d2010" if is_spike else None
            spike_fg = "#FF8C69" if is_spike else None
            self.setItem(row, 8, self._cell(f"{last:.1f}", bg=spike_bg, fg=spike_fg))

        # status
        if ps.last_error:
            self.setItem(row, 9, self._cell(ps.last_error[:22], fg="#ff8888"))
        elif ps.total_requests > 0:
            self.setItem(row, 9, self._cell("OK", fg="#a6e3a1"))
        else:
            self.setItem(row, 9, self._cell("—"))

        self.setRowHeight(row, 22)


# ── Main Window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NetPulse — HTTP Proxy Tester")
        self.resize(1200, 780)

        self._proxy_list: List[ProxyStats]      = []
        self._worker:     Optional[TestWorker]  = None
        self._color_idx   = 0

        self._build_ui()
        self._apply_dark_theme()
        self._graph_timer = QTimer(self)
        self._graph_timer.timeout.connect(self._refresh_graph)
        self._graph_timer.start(800)

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(6)
        root.setContentsMargins(8, 8, 8, 8)

        root.addWidget(self._build_toolbar())

        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)

        # top: graph + table side-by-side
        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.setChildrenCollapsible(False)

        # Left: stats table
        table_box = QGroupBox("Proxy Statistics")
        table_box.setMinimumWidth(560)
        tbl_layout = QVBoxLayout(table_box)
        tbl_layout.setContentsMargins(4, 6, 4, 4)
        self.table = StatsTable()
        tbl_layout.addWidget(self.table)
        top_splitter.addWidget(table_box)

        # Right: graph
        graph_box = QGroupBox("Real-time Graph")
        graph_layout = QVBoxLayout(graph_box)
        graph_layout.setContentsMargins(4, 6, 4, 4)
        self.graph = GraphWidget()
        graph_layout.addWidget(self.graph)
        top_splitter.addWidget(graph_box)

        top_splitter.setStretchFactor(0, 55)
        top_splitter.setStretchFactor(1, 45)
        splitter.addWidget(top_splitter)

        # bottom: proxy management
        splitter.addWidget(self._build_proxy_panel())
        splitter.setStretchFactor(0, 70)
        splitter.setStretchFactor(1, 30)

        root.addWidget(splitter)

        # status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — add proxies and press Start.")

    def _build_toolbar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(44)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        lay.addWidget(QLabel("Target URL:"))
        self.url_input = QLineEdit(DEFAULT_URL)
        self.url_input.setMinimumWidth(260)
        lay.addWidget(self.url_input)

        lay.addWidget(QLabel("Interval (s):"))
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.1, 60.0)
        self.interval_spin.setSingleStep(0.5)
        self.interval_spin.setValue(DEFAULT_INTERVAL)
        self.interval_spin.setFixedWidth(70)
        lay.addWidget(self.interval_spin)

        lay.addWidget(QLabel("Timeout (s):"))
        self.timeout_spin = QDoubleSpinBox()
        self.timeout_spin.setRange(1.0, 120.0)
        self.timeout_spin.setSingleStep(1.0)
        self.timeout_spin.setValue(DEFAULT_TIMEOUT)
        self.timeout_spin.setFixedWidth(70)
        lay.addWidget(self.timeout_spin)

        lay.addWidget(QLabel("Threads/proxy:"))
        self.concurrency_spin = QSpinBox()
        self.concurrency_spin.setRange(1, 20)
        self.concurrency_spin.setValue(DEFAULT_CONCURRENCY)
        self.concurrency_spin.setFixedWidth(55)
        lay.addWidget(self.concurrency_spin)

        lay.addStretch()

        self.btn_start = QPushButton("▶  Start")
        self.btn_start.setFixedWidth(90)
        self.btn_start.setStyleSheet("background:#1e8a3c; color:white; font-weight:bold; border-radius:4px;")
        self.btn_start.clicked.connect(self._on_start)
        lay.addWidget(self.btn_start)

        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setFixedWidth(90)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("background:#8a1e1e; color:white; font-weight:bold; border-radius:4px;")
        self.btn_stop.clicked.connect(self._on_stop)
        lay.addWidget(self.btn_stop)

        self.btn_export = QPushButton("⬇  Export CSV")
        self.btn_export.setFixedWidth(110)
        self.btn_export.clicked.connect(self._on_export)
        lay.addWidget(self.btn_export)

        self.btn_sort = QPushButton("⇅  Sort by Avg")
        self.btn_sort.setFixedWidth(110)
        self.btn_sort.clicked.connect(self._on_sort)
        lay.addWidget(self.btn_sort)

        return bar

    def _build_proxy_panel(self) -> QWidget:
        box = QGroupBox("Proxy Management  (formats:  ip:port:user:pass  |  user:pass@ip:port  |  ip:port)")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 8, 6, 6)
        lay.setSpacing(6)

        # input row
        input_row = QHBoxLayout()
        self.proxy_input = QLineEdit()
        self.proxy_input.setPlaceholderText(
            "e.g.  ip:port:user:pass  |  user:pass@ip:port  |  ip:port"
        )
        self.proxy_input.returnPressed.connect(self._on_add_proxy)
        input_row.addWidget(self.proxy_input)

        btn_add = QPushButton("Add")
        btn_add.setFixedWidth(70)
        btn_add.clicked.connect(self._on_add_proxy)
        input_row.addWidget(btn_add)

        btn_add_bulk = QPushButton("Bulk Add…")
        btn_add_bulk.setFixedWidth(90)
        btn_add_bulk.clicked.connect(self._on_bulk_add)
        input_row.addWidget(btn_add_bulk)

        btn_remove = QPushButton("Remove Selected")
        btn_remove.setFixedWidth(130)
        btn_remove.clicked.connect(self._on_remove_proxy)
        input_row.addWidget(btn_remove)

        btn_clear = QPushButton("Clear All")
        btn_clear.setFixedWidth(90)
        btn_clear.clicked.connect(self._on_clear_proxies)
        input_row.addWidget(btn_clear)

        lay.addLayout(input_row)

        self.proxy_list_widget = QListWidget()
        self.proxy_list_widget.setFixedHeight(80)
        self.proxy_list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.proxy_list_widget.setFont(QFont("Consolas", 9))
        lay.addWidget(self.proxy_list_widget)

        return box

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
            }
            QGroupBox {
                border: 1px solid #313244;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 4px;
                font-weight: bold;
                color: #89b4fa;
                font-size: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QLineEdit, QDoubleSpinBox, QSpinBox {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 3px;
                color: #cdd6f4;
                padding: 3px 6px;
                selection-background-color: #585b70;
            }
            QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus {
                border-color: #89b4fa;
            }
            QPushButton {
                background-color: #313244;
                border: 1px solid #45475a;
                border-radius: 4px;
                color: #cdd6f4;
                padding: 4px 10px;
            }
            QPushButton:hover {
                background-color: #45475a;
                border-color: #585b70;
            }
            QPushButton:disabled {
                color: #585b70;
                background-color: #1e1e2e;
            }
            QListWidget {
                background-color: #181825;
                border: 1px solid #313244;
                color: #cdd6f4;
            }
            QListWidget::item:selected {
                background-color: #45475a;
            }
            QLabel {
                color: #a6adc8;
                font-size: 9px;
            }
            QStatusBar {
                background-color: #181825;
                color: #6c7086;
                border-top: 1px solid #313244;
                font-size: 9px;
            }
            QSplitter::handle {
                background-color: #313244;
            }
        """)

    # ── Proxy management ──────────────────────────────────────────────────

    def _on_add_proxy(self):
        raw = self.proxy_input.text().strip()
        if raw:
            self._add_proxy(raw)
            self.proxy_input.clear()

    def _on_bulk_add(self):
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QTextEdit as QTE
        dlg = QDialog(self)
        dlg.setWindowTitle("Bulk Add Proxies")
        dlg.resize(420, 280)
        vl = QVBoxLayout(dlg)
        vl.addWidget(QLabel("Paste proxies, one per line:"))
        te = QTE()
        te.setFont(QFont("Consolas", 9))
        vl.addWidget(te)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        vl.addWidget(bb)
        if dlg.exec_():
            for line in te.toPlainText().splitlines():
                line = line.strip()
                if line:
                    self._add_proxy(line)

    def _add_proxy(self, raw: str):
        # avoid duplicates
        if any(ps.raw == raw for ps in self._proxy_list):
            self.status_bar.showMessage(f"Proxy already exists: {raw}")
            return
        try:
            url, auth = parse_proxy(raw)
        except Exception as exc:
            QMessageBox.warning(self, "Invalid Proxy", f"Could not parse proxy:\n{raw}\n\n{exc}")
            return

        color = PROXY_COLORS[self._color_idx % len(PROXY_COLORS)]
        self._color_idx += 1

        ps = ProxyStats(
            proxy_id=str(uuid.uuid4()),
            raw=raw,
            proxy_url=url,
            proxy_auth=auth,
            color=color,
        )
        self._proxy_list.append(ps)

        item = QListWidgetItem(f"  {ps.display_name}  ({raw})")
        item.setForeground(QBrush(QColor(color)))
        item.setData(Qt.UserRole, ps.proxy_id)
        self.proxy_list_widget.addItem(item)

        self.table.rebuild(self._proxy_list)
        self.graph.set_proxies(self._proxy_list)
        self.status_bar.showMessage(f"Added proxy: {ps.display_name}")

    def _on_remove_proxy(self):
        selected = self.proxy_list_widget.selectedItems()
        ids_to_remove = {item.data(Qt.UserRole) for item in selected}
        self._proxy_list = [ps for ps in self._proxy_list if ps.proxy_id not in ids_to_remove]
        for item in selected:
            self.proxy_list_widget.takeItem(self.proxy_list_widget.row(item))
        self.table.rebuild(self._proxy_list)
        self.graph.set_proxies(self._proxy_list)

    def _on_clear_proxies(self):
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "Running", "Stop the test before clearing proxies.")
            return
        self._proxy_list.clear()
        self._color_idx = 0
        self.proxy_list_widget.clear()
        self.table.rebuild(self._proxy_list)
        self.graph.set_proxies(self._proxy_list)

    # ── Test controls ─────────────────────────────────────────────────────

    def _on_start(self):
        if not self._proxy_list:
            QMessageBox.warning(self, "No Proxies", "Add at least one proxy before starting.")
            return

        url = self.url_input.text().strip()
        if not url.startswith("http"):
            QMessageBox.warning(self, "Invalid URL", "Target URL must start with http:// or https://")
            return

        # Reset stats
        for ps in self._proxy_list:
            ps.total_requests = 0
            ps.failures       = 0
            ps.last_error     = ""
            ps._times.clear()
            ps.history_total.clear()
            ps.history_ts.clear()
        self.table.rebuild(self._proxy_list)

        self._worker = TestWorker(
            proxy_list   = self._proxy_list,
            target_url   = url,
            interval     = self.interval_spin.value(),
            timeout      = self.timeout_spin.value(),
            concurrency  = self.concurrency_spin.value(),
        )
        self._worker.result_ready.connect(self._on_result)
        self._worker.finished.connect(self._on_worker_done)
        self._worker.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.status_bar.showMessage(
            f"Testing {len(self._proxy_list)} proxy/proxies → {url}  "
            f"(interval={self.interval_spin.value()}s  timeout={self.timeout_spin.value()}s  "
            f"threads={self.concurrency_spin.value()})"
        )

    def _on_stop(self):
        if self._worker:
            self._worker.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status_bar.showMessage("Stopped.")

    def _on_worker_done(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    # ── Result handling ───────────────────────────────────────────────────

    def _on_result(self, proxy_id: str, result: dict):
        ps = next((p for p in self._proxy_list if p.proxy_id == proxy_id), None)
        if ps is None:
            return

        if result.get("success"):
            ps.record(result["total"])
        else:
            ps.record(None, error=result.get("error", "Error"))

        self.table.update_all(self._proxy_list)

    # ── Graph refresh ─────────────────────────────────────────────────────

    def _refresh_graph(self):
        if self._proxy_list:
            self.graph.refresh()

    # ── Sort ──────────────────────────────────────────────────────────────

    def _on_sort(self):
        self._proxy_list.sort(
            key=lambda ps: ps.avg_ms if ps._times else float("inf")
        )
        self.table.rebuild(self._proxy_list)
        self.status_bar.showMessage("Sorted by average latency (fastest first).")

    # ── Export ────────────────────────────────────────────────────────────

    def _on_export(self):
        if not self._proxy_list:
            QMessageBox.information(self, "Nothing to export", "No proxy data available.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", f"netpulse_{datetime.now():%Y%m%d_%H%M%S}.csv",
            "CSV Files (*.csv)"
        )
        if not path:
            return

        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "proxy", "avg_ms", "min_ms", "max_ms",
                    "jitter_ms", "requests", "failures", "loss_pct",
                ])
                for ps in self._proxy_list:
                    writer.writerow([
                        ps.raw,
                        f"{ps.avg_ms:.2f}",
                        f"{ps.min_ms:.2f}",
                        f"{ps.max_ms:.2f}",
                        f"{ps.jitter_ms:.2f}",
                        ps.total_requests,
                        ps.failures,
                        f"{ps.loss_pct:.1f}",
                    ])
            self.status_bar.showMessage(f"Exported to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(2000)
        event.accept()


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    # Allow high-DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("NetPulse")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
