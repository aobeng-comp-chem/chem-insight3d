import numpy as np
import pyvista as pv
import os
import math
import json
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor
import vtk
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QSlider, QCheckBox, QPushButton, QLabel,
    QDockWidget, QLineEdit, QFileDialog, QListWidget, QListWidgetItem,
    QMessageBox, QButtonGroup, QRadioButton, QGroupBox, QFrame, QSpinBox,
    QDialog, QDialogButtonBox, QDoubleSpinBox, QGridLayout, QToolBar,
    QScrollArea, QSizePolicy, QStackedWidget, QAction, QMenu,
    QProgressDialog, QAbstractItemView,
)
from PyQt5.QtCore import Qt, QTimer, QSize, QPropertyAnimation, QEasingCurve, QRect, QThread, pyqtSignal
from PyQt5.QtGui import QDoubleValidator, QColor, QBrush, QFont
import sys
from pyvistaqt import QtInteractor
import matplotlib.pyplot as plt
from PIL import Image


# ── Constants ─────────────────────────────────────────────────────────────────

# ── Isosurface colour schemes ─────────────────────────────────────────────────
# Each entry: (display_name, pos_rgb_0_1, neg_rgb_0_1)
# First entry is the default.
LOBE_COLOR_SCHEMES = {
    "Mathematica (default)": ((0.60, 0.15, 0.18), (0.55, 0.72, 0.82)),   # crimson / steel-blue
    "Red / Blue":            ((0.85, 0.10, 0.10), (0.10, 0.30, 0.85)),   # classic CPK-style
    "Red / Green":           ((0.85, 0.12, 0.12), (0.12, 0.70, 0.25)),   # common in textbooks
    "Orange / Teal":         ((0.92, 0.50, 0.05), (0.05, 0.60, 0.65)),   # high-contrast warm/cool
    "Purple / Gold":         ((0.55, 0.10, 0.75), (0.85, 0.72, 0.05)),   # vivid complementary
    "White / Grey":          ((0.95, 0.95, 0.95), (0.45, 0.45, 0.45)),   # monochrome / print
}

# Active colours — initialised to the default, updated by the combo box
LOBE_POS_COLOR = LOBE_COLOR_SCHEMES["Mathematica (default)"][0]
LOBE_NEG_COLOR = LOBE_COLOR_SCHEMES["Mathematica (default)"][1]

GRID_LINES_U = 18
GRID_LINES_V = 12


# ── Atom radius scaling ───────────────────────────────────────────────────────

def space_filling_radius(vdw, cov):
    """Full VDW radius for CPK / Space Filling mode."""
    return vdw


# ── UV Grid Helper ────────────────────────────────────────────────────────────

def build_uv_grid_lines(mesh, n_u=GRID_LINES_U, n_v=GRID_LINES_V):
    if mesh is None or mesh.n_points == 0:
        return pv.PolyData()
    centre = np.array(mesh.center)
    bounds = mesh.bounds
    extents = np.array([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
    all_lines = []
    for i in range(n_u):
        angle  = math.pi * i / n_u
        normal = np.array([math.cos(angle), 0.0, math.sin(angle)])
        try:
            s = mesh.slice(normal=normal, origin=centre)
            if s.n_points > 1: all_lines.append(s)
        except Exception:
            pass
    y_min = bounds[2] + extents[1] * 0.05
    y_max = bounds[3] - extents[1] * 0.05
    for i in range(1, n_v):
        y = y_min + (y_max - y_min) * i / n_v
        try:
            s = mesh.slice(normal=[0, 1, 0], origin=[centre[0], y, centre[2]])
            if s.n_points > 1: all_lines.append(s)
        except Exception:
            pass
    if not all_lines:
        return pv.PolyData()
    combined = all_lines[0]
    for part in all_lines[1:]:
        combined = combined.merge(part)
    return combined


# ── Colour helper ─────────────────────────────────────────────────────────────

def _hex_to_rgb255(hex_color: str) -> np.ndarray:
    """Convert '#RRGGBB' to a (3,) uint8 array [R, G, B]."""
    h = hex_color.lstrip('#')
    return np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)],
                    dtype=np.uint8)


# ── Main class ────────────────────────────────────────────────────────────────

# ── NBO dialog classes (module-level, before MultiCubeVisualizer) ─────────────

class _ComputeThread(QThread):
    """Worker thread: runs compute_cube_data without blocking the GUI."""
    finished  = pyqtSignal(list)   # emits list of output cube paths
    error     = pyqtSignal(str)    # emits error message string
    progress  = pyqtSignal(str)    # emits status text

    def __init__(self, basis_path, key_path, orbital_indices, spin,
                 grid_quality, ext_dist, parent=None):
        super().__init__(parent)
        self.basis_path     = basis_path
        self.key_path       = key_path
        self.orbital_indices = orbital_indices
        self.spin           = spin
        self.grid_quality   = grid_quality
        self.ext_dist       = ext_dist

    def run(self):
        try:
            import nbo_read as _nr
            from scipy.constants import physical_constants
            bohr_const = physical_constants['Bohr radius'][0] * 1e10

            self.progress.emit("Loading and normalising basis set…")
            basis, coords, atom_info = _nr.load_basis_headless(self.basis_path)

            self.progress.emit(f"Computing {len(self.orbital_indices)} orbital(s)…")
            paths = _nr.compute_cube_data(
                basis, coords, atom_info,
                self.orbital_indices, self.key_path, self.spin,
                self.grid_quality, self.ext_dist, bohr_const
            )
            self.finished.emit(paths)
        except Exception as e:
            import traceback
            self.error.emit(traceback.format_exc())


class _OrbitalPickerDialog(QDialog):
    """
    Step 2: choose orbitals to compute from a selected key file.
    Shows a scrollable grid of checkboxes labelled 1…N.
    """
    # emits list of cube paths when done
    cubes_ready = pyqtSignal(list)

    DARK = (
        "QDialog { background:#1e1e2e; color:#cdd6f4; }"
        "QLabel  { color:#cdd6f4; }"
        "QGroupBox { color:#89b4fa; border:1px solid #313244; border-radius:4px; "
        "            margin-top:6px; padding-top:8px; }"
        "QGroupBox::title { subcontrol-origin:margin; left:8px; color:#89b4fa; }"
        "QCheckBox { color:#cdd6f4; spacing:4px; }"
        "QCheckBox::indicator { width:14px; height:14px; }"
        "QPushButton { background:#313244; color:#cdd6f4; border:1px solid #45475a; "
        "              border-radius:4px; padding:5px 12px; }"
        "QPushButton:hover   { background:#45475a; }"
        "QPushButton:disabled{ color:#6c7086; }"
        "QComboBox  { background:#313244; color:#cdd6f4; border:1px solid #45475a; padding:3px; }"
        "QSlider::groove:horizontal { background:#313244; height:4px; border-radius:2px; }"
        "QSlider::handle:horizontal { background:#89b4fa; width:12px; height:12px; "
        "                             margin:-4px 0; border-radius:6px; }"
        "QProgressDialog { background:#1e1e2e; color:#cdd6f4; }"
    )

    def __init__(self, basis_path, key_path, parent=None):
        super().__init__(parent)
        self.basis_path = basis_path
        self.key_path   = key_path
        self._thread    = None
        self._checkboxes = []

        self.setWindowTitle(f"Select Orbitals — {os.path.basename(key_path)}")
        self.setMinimumWidth(540)
        self.setStyleSheet(self.DARK)

        try:
            import nbo_read as _nr
            orb_type, nbas, is_open = _nr.get_orbital_count(key_path)
        except Exception as e:
            QMessageBox.critical(parent, "Key File Error", str(e))
            self.reject(); return

        self.nbas    = nbas
        self.is_open = is_open

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = QLabel(
            f"<b>{os.path.basename(key_path)}</b>  |  "
            f"Type: <span style='color:#89b4fa'>{orb_type}</span>  |  "
            f"Orbitals: <span style='color:#a6e3a1'>{nbas}</span>"
            + ("  |  <span style='color:#fab387'>Open shell</span>" if is_open else "")
        )
        hdr.setWordWrap(True)
        layout.addWidget(hdr)

        # ── Spin selector (open-shell only) ───────────────────────────────────
        if is_open:
            spin_row = QHBoxLayout()
            spin_row.addWidget(QLabel("Spin:"))
            self.spin_combo = QComboBox()
            self.spin_combo.addItems(["Alpha", "Beta"])
            spin_row.addWidget(self.spin_combo)
            spin_row.addStretch()
            layout.addLayout(spin_row)
        else:
            self.spin_combo = None

        # ── Orbital checkboxes ────────────────────────────────────────────────
        orb_group = QGroupBox("Orbitals  (click to select; Ctrl+A selects all)")
        orb_vlay  = QVBoxLayout(orb_group)

        # Quick-select strip
        sel_row = QHBoxLayout()
        all_btn  = QPushButton("Select All")
        none_btn = QPushButton("Select None")
        range_edit = QLineEdit(); range_edit.setPlaceholderText("e.g.  1,3,7-12")
        range_edit.setMaximumWidth(160)
        range_btn  = QPushButton("Apply Range")
        sel_row.addWidget(all_btn); sel_row.addWidget(none_btn)
        sel_row.addWidget(range_edit); sel_row.addWidget(range_btn)
        sel_row.addStretch()
        orb_vlay.addLayout(sel_row)

        # Scrollable checkbox grid
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFixedHeight(220)
        scroll.setStyleSheet("QScrollArea { border:1px solid #313244; background:#181825; }")
        inner   = QWidget()
        inner.setStyleSheet("background:#181825;")
        cols    = 10
        grid    = QGridLayout(inner)
        grid.setSpacing(2)
        for i in range(nbas):
            cb = QCheckBox(str(i + 1))
            cb.setStyleSheet("QCheckBox{color:#cdd6f4;font-size:9pt;}")
            grid.addWidget(cb, i // cols, i % cols)
            self._checkboxes.append(cb)
        scroll.setWidget(inner)
        orb_vlay.addWidget(scroll)
        layout.addWidget(orb_group)

        # Wire quick-select
        all_btn.clicked.connect(lambda: [cb.setChecked(True)  for cb in self._checkboxes])
        none_btn.clicked.connect(lambda: [cb.setChecked(False) for cb in self._checkboxes])
        def apply_range():
            txt = range_edit.text().strip()
            if not txt: return
            indices = set()
            for part in txt.split(','):
                part = part.strip()
                if '-' in part:
                    try:
                        a, b = part.split('-'); indices.update(range(int(a), int(b)+1))
                    except ValueError: pass
                else:
                    try: indices.add(int(part))
                    except ValueError: pass
            for cb in self._checkboxes: cb.setChecked(False)
            for i in indices:
                if 1 <= i <= nbas: self._checkboxes[i-1].setChecked(True)
        range_btn.clicked.connect(apply_range)

        # ── Grid settings ─────────────────────────────────────────────────────
        grid_group = QGroupBox("Grid Settings")
        gg = QGridLayout(grid_group)

        gg.addWidget(QLabel("Quality:"), 0, 0)
        self.quality_combo = QComboBox()
        self.quality_combo.addItems([
            "Low  (50 pts)", "Medium  (75 pts)",
            "Fine  (100 pts)", "Ultra-fine  (125 pts)"])
        self.quality_combo.setCurrentIndex(1)
        gg.addWidget(self.quality_combo, 0, 1)

        gg.addWidget(QLabel("Extension (bohr):"), 1, 0)
        ext_row = QHBoxLayout()
        self.ext_slider = QSlider(Qt.Horizontal)
        self.ext_slider.setMinimum(10); self.ext_slider.setMaximum(100)
        self.ext_slider.setValue(40)
        self.ext_label  = QLabel("4.0")
        self.ext_slider.valueChanged.connect(
            lambda v: self.ext_label.setText(f"{v/10:.1f}"))
        ext_row.addWidget(self.ext_slider); ext_row.addWidget(self.ext_label)
        ext_widget = QWidget(); ext_widget.setLayout(ext_row)
        gg.addWidget(ext_widget, 1, 1)

        layout.addWidget(grid_group)

        # ── Progress / status ─────────────────────────────────────────────────
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color:#a6adc8; font-size:9pt;")
        layout.addWidget(self.status_label)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.compute_btn = QPushButton("▶  Compute Selected Orbitals")
        self.compute_btn.setStyleSheet(
            "QPushButton { background:#89b4fa; color:#1e1e2e; font-weight:bold; "
            "              border-radius:4px; padding:6px 16px; }"
            "QPushButton:hover { background:#b4d0fa; }"
            "QPushButton:disabled { background:#45475a; color:#6c7086; }")
        cancel_btn = QPushButton("Cancel")
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(self.compute_btn)
        layout.addLayout(btn_row)

        self.compute_btn.clicked.connect(self._start_compute)
        cancel_btn.clicked.connect(self.reject)

    def _selected_indices(self):
        return [i+1 for i, cb in enumerate(self._checkboxes) if cb.isChecked()]

    def _start_compute(self):
        indices = self._selected_indices()
        if not indices:
            QMessageBox.warning(self, "Nothing selected", "Please select at least one orbital.")
            return

        quality_map = {0: 50, 1: 75, 2: 100, 3: 125}
        grid_quality = quality_map[self.quality_combo.currentIndex()]
        ext_dist     = self.ext_slider.value() / 10.0
        spin = "beta" if (self.spin_combo and self.spin_combo.currentIndex() == 1) else "alpha"

        self.compute_btn.setEnabled(False)
        self.status_label.setText(
            f"Starting computation for {len(indices)} orbital(s)…")

        self._thread = _ComputeThread(
            self.basis_path, self.key_path, indices, spin,
            grid_quality, ext_dist, parent=self)
        self._thread.progress.connect(self.status_label.setText)
        self._thread.finished.connect(self._on_done)
        self._thread.error.connect(self._on_error)
        self._thread.start()

    def _on_done(self, paths):
        self.compute_btn.setEnabled(True)
        self.status_label.setText(
            f"✓  {len(paths)} cube file(s) written.")
        self.cubes_ready.emit(paths)
        self.accept()

    def _on_error(self, msg):
        self.compute_btn.setEnabled(True)
        self.status_label.setText("✗  Computation failed.")
        QMessageBox.critical(self, "Computation Error", msg)


class _KeyFilePickerDialog(QDialog):
    """
    Step 1: given a basis file path, show all sibling key files (same stem,
    different extension) and let the user pick one to proceed with.
    """
    cubes_ready = pyqtSignal(list)

    DARK = _OrbitalPickerDialog.DARK  # reuse same stylesheet

    # Extensions we know are NBO key files (exclude cube, basis, and plain text)
    _BASIS_EXTS = {'.47', '.31', '.32', '.33'}
    _SKIP_EXTS  = {'.cube', '.log', '.out', '.txt', '.py',
                   '.json', '.png', '.pdf', '.svg'}

    def __init__(self, basis_path, parent=None):
        super().__init__(parent)
        self.basis_path = basis_path
        self.setWindowTitle("Select Key File")
        self.setMinimumWidth(500)
        self.setStyleSheet(self.DARK)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        stem    = os.path.splitext(basis_path)[0]
        dirpath = os.path.dirname(basis_path) or '.'
        base    = os.path.basename(stem)

        layout.addWidget(QLabel(
            f"Basis file: <b>{os.path.basename(basis_path)}</b><br>"
            f"Looking for key files with stem <b>{base}</b> …"
        ))

        # Find all files with same stem, exclude basis and noise extensions
        import glob
        candidates = sorted(glob.glob(os.path.join(dirpath, base + '.*')))
        key_files  = []
        for p in candidates:
            ext = os.path.splitext(p)[1].lower()
            if ext in self._BASIS_EXTS or ext in self._SKIP_EXTS:
                continue
            key_files.append(p)

        if not key_files:
            layout.addWidget(QLabel(
                "<span style='color:#f38ba8;'>No key files found next to the basis file.<br>"
                "Make sure the NBO key files (.31, .32, .33 etc.) are in the same folder.</span>"))
            ok = QPushButton("OK"); ok.clicked.connect(self.reject)
            layout.addWidget(ok)
            return

        layout.addWidget(QLabel(
            f"Found <b>{len(key_files)}</b> key file(s). "
            "Double-click one to select orbitals:"))

        # List widget
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(
            "QListWidget { background:#181825; border:1px solid #313244; "
            "              color:#cdd6f4; font-size:10pt; }"
            "QListWidget::item { padding:6px 8px; border-bottom:1px solid #313244; }"
            "QListWidget::item:selected { background:#89b4fa; color:#1e1e2e; font-weight:bold; }"
            "QListWidget::item:hover { background:#313244; }")
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)

        import nbo_read as _nr
        for path in key_files:
            try:
                orb_type, nbas, is_open = _nr.get_orbital_count(path)
                tag  = "  [open shell]" if is_open else ""
                text = f"{os.path.basename(path)}    ·  {orb_type}  ·  {nbas} orbitals{tag}"
            except Exception:
                text = os.path.basename(path)
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, path)
            self.list_widget.addItem(item)

        self.list_widget.setCurrentRow(0)
        self.list_widget.itemDoubleClicked.connect(self._pick_item)
        layout.addWidget(self.list_widget)

        btn_row = QHBoxLayout()
        select_btn = QPushButton("Select →")
        select_btn.setStyleSheet(
            "QPushButton { background:#89b4fa; color:#1e1e2e; font-weight:bold; "
            "              border-radius:4px; padding:6px 16px; }"
            "QPushButton:hover { background:#b4d0fa; }")
        cancel_btn = QPushButton("Cancel")
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(select_btn)
        layout.addLayout(btn_row)

        select_btn.clicked.connect(self._pick_selected)
        cancel_btn.clicked.connect(self.reject)

    def _pick_item(self, item):
        self._open_orbital_dialog(item.data(Qt.UserRole))

    def _pick_selected(self):
        item = self.list_widget.currentItem()
        if item:
            self._open_orbital_dialog(item.data(Qt.UserRole))

    def _open_orbital_dialog(self, key_path):
        dlg = _OrbitalPickerDialog(self.basis_path, key_path, self)
        dlg.cubes_ready.connect(self._relay)
        dlg.exec_()

    def _relay(self, paths):
        self.cubes_ready.emit(paths)
        self.accept()



class MultiCubeVisualizer:
    # (symbol, vdw_radius_Å, cov_radius_Å, bs_display_radius_Å, cpk_color)
    #
    # cov_radius  – Alvarez 2008 / Pyykkö 2009 covalent radii, used ONLY for
    #               bond detection (threshold = 1.15 × sum of cov radii)
    # vdw_radius  – Bondi / Alvarez 2013 VDW radii, used ONLY for Space Filling
    # bs_radius   – uniform per periodic-table row for Ball & Stick display:
    #               Row 1: 0.18  Row 2: 0.22  Row 3: 0.26
    #               Row 4: 0.32  Row 5: 0.36  Row 6: 0.40  Row 7: 0.44
    # Colors      – Jmol / CPK2 color scheme
    ELEMENT_DATA = {
        # Z    sym    vdw    cov   bs_r   color
        # ── Row 1 ──────────────────────────────────────────────────────────
        1:  ('H',  1.20, 0.31, 0.18, '#FFFFFF'),
        2:  ('He', 1.40, 0.28, 0.18, '#D9FFFF'),
        # ── Row 2 ──────────────────────────────────────────────────────────
        3:  ('Li', 1.82, 1.28, 0.22, '#CC80FF'),
        4:  ('Be', 1.53, 0.96, 0.22, '#C2FF00'),
        5:  ('B',  1.92, 0.85, 0.22, '#FFB5B5'),
        6:  ('C',  1.70, 0.76, 0.22, '#909090'),
        7:  ('N',  1.55, 0.71, 0.22, '#3050F8'),
        8:  ('O',  1.52, 0.66, 0.22, '#FF0D0D'),
        9:  ('F',  1.47, 0.57, 0.22, '#90E050'),
        10: ('Ne', 1.54, 0.58, 0.22, '#B3E3F5'),
        # ── Row 3 ──────────────────────────────────────────────────────────
        11: ('Na', 2.27, 1.66, 0.26, '#AB5CF2'),
        12: ('Mg', 1.73, 1.41, 0.26, '#8AFF00'),
        13: ('Al', 1.84, 1.21, 0.26, '#BFA6A6'),
        14: ('Si', 2.10, 1.11, 0.26, '#F0C8A0'),
        15: ('P',  1.80, 1.07, 0.26, '#FF8000'),
        16: ('S',  1.80, 1.05, 0.26, '#FFFF30'),
        17: ('Cl', 1.75, 1.02, 0.26, '#1FF01F'),
        18: ('Ar', 1.88, 1.06, 0.26, '#80D1E3'),
        # ── Row 4 (K–Kr, 3d block) ─────────────────────────────────────────
        19: ('K',  2.75, 2.03, 0.32, '#8F40D4'),
        20: ('Ca', 2.31, 1.76, 0.32, '#3DFF00'),
        21: ('Sc', 2.11, 1.70, 0.32, '#E6E6E6'),
        22: ('Ti', 1.87, 1.60, 0.32, '#BFC2C7'),
        23: ('V',  1.79, 1.53, 0.32, '#A6A6AB'),
        24: ('Cr', 1.89, 1.39, 0.32, '#8A99C7'),
        25: ('Mn', 1.97, 1.61, 0.32, '#9C7AC7'),
        26: ('Fe', 1.94, 1.52, 0.32, '#E06633'),
        27: ('Co', 1.92, 1.50, 0.32, '#F090A0'),
        28: ('Ni', 1.84, 1.24, 0.32, '#50D050'),
        29: ('Cu', 1.86, 1.32, 0.32, '#C88033'),
        30: ('Zn', 2.10, 1.22, 0.32, '#7D80B0'),
        31: ('Ga', 1.87, 1.22, 0.32, '#C28F8F'),
        32: ('Ge', 2.11, 1.20, 0.32, '#668F8F'),
        33: ('As', 1.85, 1.19, 0.32, '#BD80E3'),
        34: ('Se', 1.90, 1.20, 0.32, '#FFA100'),
        35: ('Br', 1.85, 1.20, 0.32, '#A62929'),
        36: ('Kr', 2.02, 1.16, 0.32, '#5CB8D1'),
        # ── Row 5 (Rb–Xe, 4d block) ────────────────────────────────────────
        37: ('Rb', 3.03, 2.20, 0.36, '#702EB0'),
        38: ('Sr', 2.49, 1.95, 0.36, '#00FF00'),
        39: ('Y',  2.32, 1.90, 0.36, '#94FFFF'),
        40: ('Zr', 2.23, 1.75, 0.36, '#94E0E0'),
        41: ('Nb', 2.18, 1.64, 0.36, '#73C2C9'),
        42: ('Mo', 2.17, 1.54, 0.36, '#54B5B5'),
        43: ('Tc', 2.16, 1.47, 0.36, '#3B9E9E'),
        44: ('Ru', 2.13, 1.46, 0.36, '#248F8F'),
        45: ('Rh', 2.10, 1.42, 0.36, '#0A7D8C'),
        46: ('Pd', 2.10, 1.39, 0.36, '#006985'),
        47: ('Ag', 2.11, 1.45, 0.36, '#C0C0C0'),
        48: ('Cd', 2.18, 1.44, 0.36, '#FFD98F'),
        49: ('In', 1.93, 1.42, 0.36, '#A67573'),
        50: ('Sn', 2.17, 1.39, 0.36, '#668080'),
        51: ('Sb', 2.06, 1.39, 0.36, '#9E63B5'),
        52: ('Te', 2.06, 1.38, 0.36, '#D47A00'),
        53: ('I',  1.98, 1.39, 0.36, '#940094'),
        54: ('Xe', 2.16, 1.40, 0.36, '#429EB0'),
        # ── Row 6 (Cs–Rn, 4f lanthanides + 5d block) ──────────────────────
        55: ('Cs', 3.43, 2.44, 0.40, '#57178F'),
        56: ('Ba', 2.68, 2.15, 0.40, '#00C900'),
        57: ('La', 2.43, 2.07, 0.40, '#70D4FF'),
        58: ('Ce', 2.42, 2.04, 0.40, '#FFFFC7'),
        59: ('Pr', 2.40, 2.03, 0.40, '#D9FFC7'),
        60: ('Nd', 2.39, 2.01, 0.40, '#C7FFC7'),
        61: ('Pm', 2.38, 1.99, 0.40, '#A3FFC7'),
        62: ('Sm', 2.36, 1.98, 0.40, '#8FFFC7'),
        63: ('Eu', 2.35, 1.98, 0.40, '#61FFC7'),
        64: ('Gd', 2.34, 1.96, 0.40, '#45FFC7'),
        65: ('Tb', 2.33, 1.94, 0.40, '#30FFC7'),
        66: ('Dy', 2.31, 1.92, 0.40, '#1FFFC7'),
        67: ('Ho', 2.30, 1.92, 0.40, '#00FF9C'),
        68: ('Er', 2.29, 1.89, 0.40, '#00E675'),
        69: ('Tm', 2.27, 1.90, 0.40, '#00D452'),
        70: ('Yb', 2.26, 1.87, 0.40, '#00BF38'),
        71: ('Lu', 2.24, 1.87, 0.40, '#00AB24'),
        72: ('Hf', 2.23, 1.75, 0.40, '#4DC2FF'),
        73: ('Ta', 2.22, 1.70, 0.40, '#4DA6FF'),
        74: ('W',  2.18, 1.62, 0.40, '#2194D6'),
        75: ('Re', 2.16, 1.51, 0.40, '#267DAB'),
        76: ('Os', 2.16, 1.44, 0.40, '#266696'),
        77: ('Ir', 2.13, 1.41, 0.40, '#175487'),
        78: ('Pt', 2.13, 1.36, 0.40, '#D0D0E0'),
        79: ('Au', 2.14, 1.36, 0.40, '#FFD123'),
        80: ('Hg', 2.23, 1.32, 0.40, '#B8B8D0'),
        81: ('Tl', 1.96, 1.45, 0.40, '#A6544D'),
        82: ('Pb', 2.02, 1.46, 0.40, '#575961'),
        83: ('Bi', 2.07, 1.48, 0.40, '#9E4FB5'),
        84: ('Po', 1.97, 1.40, 0.40, '#AB5C00'),
        85: ('At', 2.02, 1.50, 0.40, '#754F45'),
        86: ('Rn', 2.20, 1.50, 0.40, '#428296'),
        # ── Row 7 (Fr–Og, 5f actinides + 6d block) ─────────────────────────
        87: ('Fr', 3.48, 2.60, 0.44, '#420066'),
        88: ('Ra', 2.83, 2.21, 0.44, '#007D00'),
        89: ('Ac', 2.47, 2.15, 0.44, '#70ABFA'),
        90: ('Th', 2.45, 2.06, 0.44, '#00BAFF'),
        91: ('Pa', 2.43, 2.00, 0.44, '#00A1FF'),
        92: ('U',  2.41, 1.96, 0.44, '#008FFF'),
        93: ('Np', 2.39, 1.90, 0.44, '#0080FF'),
        94: ('Pu', 2.43, 1.87, 0.44, '#006BFF'),
        95: ('Am', 2.44, 1.80, 0.44, '#545CF2'),
        96: ('Cm', 2.45, 1.69, 0.44, '#785CE3'),
        97: ('Bk', 2.44, 1.68, 0.44, '#8A4FE3'),
        98: ('Cf', 2.45, 1.68, 0.44, '#A136D4'),
        99: ('Es', 2.45, 1.65, 0.44, '#B31FD4'),
        100:('Fm', 2.45, 1.67, 0.44, '#B31FBA'),
        101:('Md', 2.46, 1.73, 0.44, '#B30DA6'),
        102:('No', 2.46, 1.76, 0.44, '#BD0D87'),
        103:('Lr', 2.46, 1.61, 0.44, '#C70066'),
        104:('Rf', 2.00, 1.57, 0.44, '#CC0059'),
        105:('Db', 2.00, 1.49, 0.44, '#D1004F'),
        106:('Sg', 2.00, 1.43, 0.44, '#D90045'),
        107:('Bh', 2.00, 1.41, 0.44, '#E00038'),
        108:('Hs', 2.00, 1.34, 0.44, '#E6002E'),
        109:('Mt', 2.00, 1.29, 0.44, '#EB0026'),
        110:('Ds', 2.00, 1.28, 0.44, '#F00021'),
        111:('Rg', 2.00, 1.21, 0.44, '#F5001A'),
        112:('Cn', 2.00, 1.22, 0.44, '#F90014'),
        113:('Nh', 2.00, 1.36, 0.44, '#FC000E'),
        114:('Fl', 2.00, 1.43, 0.44, '#FE0008'),
        115:('Mc', 2.00, 1.62, 0.44, '#FF0003'),
        116:('Lv', 2.00, 1.75, 0.44, '#FF0000'),
        117:('Ts', 2.00, 1.65, 0.44, '#FA0000'),
        118:('Og', 2.00, 1.57, 0.44, '#F50000'),
    }

    MAX_SELECTION = 4

    # -------------------------------------------------------------------------
    # Molecule themes
    #
    # Ball and Stick:
    #   Atoms  → compressed cov^0.6 scaling (VMD-style, reduces H/C size gap)
    #   Bonds  → 0.10 Å cylinders
    #
    # Sticks:
    #   No atom spheres at all.  Instead each bond cylinder is extended by
    #   half-a-radius on each end so it reaches cleanly into the atom centre.
    #   Result: a pure stick representation with no visible end-cap artefacts.
    #
    # Space Filling:
    #   Atoms  → full VDW radius (CPK model), no bonds
    # -------------------------------------------------------------------------
    MOLECULE_THEMES = {
        "Ball and Stick": dict(
            mode='ball_and_stick',
            bond_radius_A=0.10,
        ),
        "Sticks": dict(
            mode='sticks',
            bond_radius_A=0.10,
        ),
        "Space Filling": dict(
            mode='space_filling',
            bond_radius_A=0.00,
        ),
    }

    def __init__(self, cube_files=None):
        self.cube_files = list(cube_files) if cube_files else []
        self.current_cube_index = 0
        self.cubes = []
        for fname in self.cube_files:
            self.cubes.append(self.read_cube(fname))
        # (no ValueError if empty — user can drag files in later)

        self.current_isovalue  = 0.03
        self.current_mol_theme = "Ball and Stick"
        self.surface_opacity   = 1.0
        self.show_wireframe    = True
        self.show_atom_labels  = False
        self.background_color  = 'black'

        self.lobe_pos_color  = LOBE_COLOR_SCHEMES["Mathematica (default)"][0]
        self.lobe_neg_color  = LOBE_COLOR_SCHEMES["Mathematica (default)"][1]
        self.color_scheme_combo = None

        self.pos_actor       = None
        self.neg_actor       = None
        self.pos_wire_actor  = None
        self.neg_wire_actor  = None
        self.pos_mesh        = None
        self.neg_mesh        = None
        self.atom_actors     = []
        self.bond_actors     = []
        self.title_actor     = None
        self.atom_label_actors        = []
        self.selected_atoms           = []
        self.selected_atom_highlights = []

        self.plotter                    = None
        self.app                        = None
        self.main_window                = None
        self.isovalue_slider            = None
        self.isovalue_edit              = None
        self.opacity_slider             = None
        self.opacity_label_value        = None
        self.selected_atoms_list_widget = None
        self.selection_label            = None
        self.measurement_label          = None
        self.format_combo               = None
        self.resolution_combo           = None
        self.transp_bg_check            = None
        self.cube_list                  = None

        # Feature 2 — metadata panel
        self.metadata_label             = None


        # Feature 4 — camera presets
        self.camera_preset_btns         = []

        # Session state
        self.session_file               = None

        # Open panel references (non-modal dialogs)
        self._panels                    = {}

        # SSAO
        self.ssao_enabled               = False

        # Viewport measurement annotation actors
        self.measurement_actors         = []

        # Animation
        self.anim_timer                 = None
        self.anim_playing               = False
        self.anim_spf                   = 1.0   # seconds per frame (default 1 s)
        self.play_btn                   = None
        self.spf_slider                 = None
        self.spf_label                  = None

    # ── Cube I/O ──────────────────────────────────────────────────────────────

    def read_cube(self, fname):
        with open(fname) as f:
            f.readline(); f.readline()
            parts  = f.readline().split()
            natoms = int(parts[0])
            origin = np.array([float(x) for x in parts[1:4]])
            nx, *xvec = [float(x) for x in f.readline().split()]
            ny, *yvec = [float(x) for x in f.readline().split()]
            nz, *zvec = [float(x) for x in f.readline().split()]
            unit_bohr = (xvec[0] > 0)
            atoms, coords, coord_lines = [], [], []
            for _ in range(int(natoms)):
                p  = f.readline().split()
                an = abs(int(p[0]))
                atoms.append(an)
                c  = [float(x) for x in p[2:5]]
                coords.append(c)
                coord_lines.append([an] + c)
            data = []
            for line in f:
                data.extend(float(x) for x in line.split())
        nx, ny, nz = int(nx), int(ny), int(nz)
        if len(data) != nx * ny * nz:
            raise ValueError(f"Data size mismatch in {fname}")
        self.unit_bohr = unit_bohr
        bonds = self._detect_bonds(coord_lines, unit_bohr)
        return dict(
            origin=origin,
            spacing=(xvec[0], yvec[1], zvec[2]),
            dimensions=(nx, ny, nz),
            atoms=np.array(atoms),
            coordinates=np.array(coords),
            data=np.array(data).reshape((nx, ny, nz)),
            bonds=bonds,
            unit_bohr=unit_bohr,
        )

    def _detect_bonds(self, coord_lines, unit_bohr):
        atoms = [{'an': p[0], 'xyz': (p[1], p[2], p[3])} for p in coord_lines]
        bonds = []
        for i, j in combinations(range(len(atoms)), 2):
            a1, a2 = atoms[i], atoms[j]
            e1 = self.ELEMENT_DATA.get(a1['an'])
            e2 = self.ELEMENT_DATA.get(a2['an'])
            if not e1 or not e2:
                continue
            thresh = 1.15 * (e1[2] + e2[2])
            if unit_bohr:
                thresh /= 0.529177
            x1, y1, z1 = a1['xyz']; x2, y2, z2 = a2['xyz']
            d = math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
            if d <= thresh:
                bonds.append(dict(an1=a1['an'], an2=a2['an'],
                                  c1=a1['xyz'], c2=a2['xyz']))
        return bonds

    # ── Grid ────â───────────────────────────────────────────────

    def _create_grid(self):
        cube = self.cubes[self.current_cube_index]
        grid = pv.ImageData(dimensions=cube['dimensions'],
                            spacing=cube['spacing'],
                            origin=cube['origin'])
        grid.point_data['values'] = cube['data'].flatten(order='F')
        return grid

    # ── Atom radius (native un ─────────────────────────────────────────────

    

    def _atom_display_radius(self, atomic_num):
        """Return display radius in cube-file native units for current theme."""
        cube = self.cubes[self.current_cube_index]
        b2a  = 0.529177 if cube['unit_bohr'] else 1.0
        e    = self.ELEMENT_DATA.get(atomic_num)
        if not e:
            return 0.35 / b2a
        vdw, cov, bs_r = e[1], e[2], e[3]
        mode = self.MOLECULE_THEMES[self.current_mol_theme]['mode']
        if mode == 'ball_and_stick':
            r_ang = bs_r           # hand-tuned per-row display radius
        elif mode == 'space_filling':
            r_ang = space_filling_radius(vdw, cov)   # true VDW radius
        else:
            r_ang = 0.0
        return r_ang / b2a

    # ── Lobe rendering ────────────────────────────────────────────────────

    def _add_lobe(self, mesh, color):
        return self.plotter.add_mesh(
            mesh,
            color=color,
            opacity=self.surface_opacity,
            style='surface',
            smooth_shading=True,
            specular=0.4, specular_power=20,
            diffuse=0.85, ambient=0.15,
            show_scalar_bar=False,
        )

    def _add_wireframe_overlay(self, mesh):
        grid_lines = build_uv_grid_lines(mesh)
        if grid_lines.n_points == 0:
            return self.plotter.add_mesh(
            grid_lines,
            style='wireframe',
            color=(0.08, 0.08, 0.08),
            line_width=0.8,
            show_scalar_bar=False,
            render_lines_as_tubes=False,
            opacity=0.75,
        )

    # ── Atoms ─────────────────────────────────────────────────────────────────

    def _add_atoms(self):
        """
        Render all atoms as a single merged mesh with a per-point RGB color array.
        One add_mesh call instead of N — dramatically faster for large systems.
        """
        cube  = self.cubes[self.current_cube_index]
        mode  = self.MOLECULE_THEMES[self.current_mol_theme]['mode']
        if mode == 'sticks':
            return
        b2a = 0.529177 if cube['unit_bohr'] else 1.0

        meshes = []
        colors = []
        RES = 24

        for an, pos in zip(cube['atoms'], cube['coordinates']):
            e = self.ELEMENT_DATA.get(an)
            if not e:
                continue
            symbol, vdw, cov, bs_r, color = e
            r = (vdw if mode == 'space_filling' else bs_r) / b2a
            sphere = pv.Sphere(radius=r, center=pos,
                               theta_resolution=RES, phi_resolution=RES)
            rgb = _hex_to_rgb255(color)
            # Attach color to each sphere before merging so points are tagged
            sphere.point_data['colors'] = np.full((sphere.n_points, 3),
                                                   rgb, dtype=np.uint8)
            meshes.append(sphere)

        if not meshes:
            return

        # merge_points=False: keep every point as-is, no deduplication
        combined = pv.merge(meshes, merge_points=False)

        actor = self.plotter.add_mesh(
            combined,
            scalars='colors',
            rgb=True,
            smooth_shading=True,
            specular=0.5, specular_power=40,
            diffuse=0.85, ambient=0.08,
            show_scalar_bar=False,
        )
        self.atom_actors.append(actor)

    # ── Bonds ─────────────────────────────────────────────────────────────────

    def _add_bonds(self):
        """
        Render all bond cylinders as a single merged mesh with per-point colors.
        One add_mesh call instead of 2xN — fast for large systems.
        """
        cube  = self.cubes[self.current_cube_index]
        theme = self.MOLECULE_THEMES[self.current_mol_theme]
        brad  = theme['bond_radius_A']
        if brad <= 0:
            return
        b2a  = 0.529177 if cube['unit_bohr'] else 1.0
        r    = brad / b2a
        mode = theme['mode']

        meshes = []

        for bond in cube['bonds']:
            e1 = self.ELEMENT_DATA.get(bond['an1'])
            e2 = self.ELEMENT_DATA.get(bond['an2'])
            if not e1 or not e2:
                continue
            cov1, col1 = e1[2], e1[4]
            cov2, col2 = e2[2], e2[4]
            frac = cov1 / (cov1 + cov2)
            p1   = np.array(bond['c1']); p2 = np.array(bond['c2'])
            mid  = p1 + (p2 - p1) * frac

            if mode == 'sticks':
                direction = p2 - p1
                length    = np.linalg.norm(direction)
                if length < 1e-6: continue
                unit = direction / length
                ext  = r * 1.5
                segs = [(p1 - unit*ext, mid, col1),
                        (mid, p2 + unit*ext, col2)]
            else:
                segs = [(p1, mid, col1), (mid, p2, col2)]

            for start, end, col in segs:
                if np.linalg.norm(end - start) < 1e-6:
                    continue
                tube = pv.Line(start, end).tube(radius=r, n_sides=12)
                rgb  = _hex_to_rgb255(col)
                tube.point_data['colors'] = np.full((tube.n_points, 3),
                                                     rgb, dtype=np.uint8)
                meshes.append(tube)

        if not meshes:
            return

        combined = pv.merge(meshes, merge_points=False)
        actor = self.plotter.add_mesh(
            combined, scalars='colors', rgb=True,
            smooth_shading=True,
            specular=0.3, specular_power=20,
            diffuse=0.9, ambient=0.08,
            show_scalar_bar=False,
        )
        self.bond_actors.append(actor)

    # ── Atom labels ───────────────────────────────────────────────────────

    def _add_atom_labels(self):
        """
        Build all billboard label actors in parallel (ThreadPoolExecutor),
        then add them to the renderer in one sequential pass on the main thread.

        VTK actor construction is pure Python object work — no OpenGL calls —
        so it is safe to do in worker threads.  Only AddActor() must stay on
        the main thread (OpenGL ctext requirement).
        """
        cube     = self.cubes[self.current_cube_index]
        renderer = self.plotter.renderer
        camera   = renderer.GetActiveCamera()
        b2a      = 0.529177 if cube['unit_bohr'] else 1.0

        fg        = (0.05, 0.05, 0.05) if self.background_color == 'white' else (1.0, 1.0, 1.0)
        FONT_SIZE = 13
        MARGIN_PX = 5
        mode      = self.MOLECULE_THEMES[self.current_mol_theme]['mode']

        # Snapshot camera-up once (thread-safe read)
        up = np.array(camera.GetViewUp())
        norm = np.linalg.norm(up)
        up   = up / norm if norm > 1e-6 else np.array([0.0, 1.0, 0.0])

        # Pre-compute world→display for ALL atoms using a single vtkCoordinate
        # (must stay on main thread — VTK coordinate transform touches renderer)
        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToWorld()

        def world_to_display(pt):
            coord.SetValue(float(pt[0]), float(pt[1]), float(pt[2]))
            return coord.GetCtedDisplayValue(renderer)

        # Build per-atom parameter tuples (all main-thread, fast)
        atom_params = []
        for i, (an, pos) in enumerate(zip(cube['atoms'], cube['coordinates'])):
            e      = self.ELEMENT_DATA.get(an)
            symbol = e[0] if e else f"X{an}"
            vdw    = e[1] if e else 1.2
            bs_r   = e[3] if e else 0.22
            r_world = (vdw if mode == "space_filling" else bs_r) / b2a
            centre_px = world_to_display(pos)
            top_px    = world_to_display(np.array(pos) + up * r_world)
            screen_r  = abs(top_px[1] - centre_px[1])
            offset_y  = max(int(screen_r + MARGIN_PX), 8)
            atom_params.append((i, symbol, tuple(pos), offset_y))

        # Worker: construct one vtkBillboardTextActor3D (no OpenGL, thread-safe)
        def build_actor(params):
            i, symbol, pos, offset_y = params
            label = f"{symbol}{i+1}"
            actor = vtk.vtkBillboardTextActor3D()
            actor.SetInput(label)
            actor.SetPosition(*pos)
            actor.SetDisplayOffset(0, offset_y)
            tp = actor.GetTextProperty()
            tp.SetFontFamilyToArial()
            tp.SetFontSize(FONT_SIZE)
            tp.SetBold(True)
            tp.SetItalic(False)
            tp.SetColor(*fg)
            tp.SetShadow(True)
            tp.SetShadowOffset(1, -1)
            tp.SetJustificationToCentered()
            tp.SetVerticalJustificationToBottom()
            return actor

        # Build actors in parallel âesults come back in submission order
        n_workers = min(8, len(atom_params))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            actors = list(pool.map(build_actor, atom_params))

        # Add to renderer on main thread (OpenGL)
        for actor in actors:
            renderer.AddActor(actor)
            self.atom_label_actors.append(actor)

    # ── Title ───────────────────────────────────────────────────────────────

    def _title_color(self):
        return 'black' if self.background_color == 'white' else 'white'

    def _update_title(self):
        if self.title_actor:
            self.plotter.remove_actor(self.title_actor); self.title_actor = None
        text = (
            f"{os.path.basename(self.cube_files[self.current_cube_index])}\n"
            f"Isovalue: \u00b1{self.current_isovalue:.4f}  |  "
            f"{self.current_mol_theme}  |  Opacity: {self.surface_opacity:.2f}"
        )
        self.title_actor = self.plotter.add_text(
            text, position='upper_right', font_size=11, color=self._title_color()
        )

    # ── Master update ─────────────────────────────────────────────────────────

    def _clear_actors(self):
        for attr in ('pos_actor', 'neg_actor', 'pos_wire_actor',
                     'ne_actor', 'title_actor'):
            a = getattr(self, attr)
            if a:
                self.plotter.remove_actor(a); setattr(self, attr, None)
        for a in self.atom_actors + self.bond_actors + self.atom_label_actors:
            self.plotter.remove_actor(a)
        self.atom_actors = []; self.bond_actors = []; self.atom_label_actors = []
        for a in self.selected_atom_highlights:
            self.plotter.remove_actor(a)
        self.selected_atom_highlights = []

    def update_visualization(self):
        self._clear_actors()
        grid          = self._create_grid()
        self.pos_mesh = grid.contour([self.current_isovalue])
        self.neg_mesh = grid.contour([-self.current_isovalue])
        self.pos_actor = self._add_lobe(self.pos_mesh, self.lobe_pos_color)
        self.neg_actor = self._add_lobe(self.neg_mesh, self.lobe_neg_color)
        if self.show_wireframe:
            self.pos_wire_actor = self._add_wireframe_overlay(self.pos_mesh)
            self.neg_wire_actor = self._add_wireframe_overlay(self.neg_mesh)
        self._add_bonds()
        self._add_atoms()
        if self.show_atom_labels:
            self._add_atom_labels()
        self._update_selected_highlights()
        self._update_title()
        self.refresh_metadata()
        self.plotter.render()

    # ── Isovalue ──────────────────────────────────────────────────────────────

    def update_isovalue(self, value):
        if isinstance(value, str):
            try: value = float(value)
            except ValueError: return
        max_val = max(c['data'].max() for c in self.cubes)
        value   = float(np.clip(value, 1e-5, max_val))
        self.current_isovalue = value
        if self.isovalue_edit:
            try:
                if abs(float(self.isovalue_edit.text()) - value) > 1e-5:
                    self.isovalue_edit.setText(f"{value:.4f}")
            except ValueError:
                self.isovalue_edit.setText(f"{value:.4f}")
        if self.isovalue_slider:
            self.isovalue_slider.blockSignals(True)
            self.isovalue_slider.setValue(int(value * 10000))
            self.isovalue_slider.blockSignals(False)
        for attr in ('pos_actor', 'neg_actor', 'pos_wire_actor', 'neg_wire_actor'):
            a = getattr(self, attr)
            if a: self.plotter.remove_actor(a); setattr(self, attr, None)
        grid          = self._create_grid()
        self.pos_mesh = grid.contour([value])
        self.neg_mesh = grid.contour([-value])
        self.pos_actor = self._add_lobe(self.pos_mesh, self.lobe_pos_color)
        self.neg_actor = self._add_lobe(self.neg_mesh, self.lobe_neg_color)
        if self.show_wireframe:
            self.pos_wire_actor = self._add_wireframe_overlay(self.pos_mesh)
            self.neg_wire_actor = self._add_wireframe_overlay(self.neg_mesh)
        self._update_title()
        self.refresh_metadata()
        self.plotter.render()

    # ─opacity ───────────────────────────────────────────────────────────────

   

    def update_opacity(self, slider_value):
        opacity = slider_value / 100.0
        self.surface_opacity = opacity
        if self.opacity_label_value:
            self.opacity_label_value.setText(f"{opacity:.2f}")
        for actor in (self.pos_actor, self.neg_actor):
            if actor:
                actor.GetProperty().SetOpacity(opacity)
        self._update_title()
        self.plotter.render()

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def set_molecule_theme(self, theme_name):
        self.current_mol_theme = theme_name
        self.update_visualization()

    def set_lobe_colors(self, scheme_name: str):
        """Switch isosurface colour pair a the lobe actors."""
        pos, neg = LOBE_COLOR_SCHEMES[scheme_name]
        self.lobe_pos_color = pos
        self.lobe_neg_color = neg

        # Swap actors without rebuilding the whole scene
        for attr in ('pos_actor', 'neg_actor', 'pos_wire_actor', 'neg_wire_actor'):
            a = getattr(self, attr)
            if a: self.plotter.remove_actor(a); setattr(self, attr, None)

        if self.pos_mesh:
            self.pos_actor = self._add_lobe(self.pos_mesh, self.lobe_pos_color)
        if self.neg_mesh:
            self.neg_actor = self._add_lobe(self.neg_mesh, self.lobe_neg_color)
        if self.show_wireframe:
            if self.pos_mesh:
                self.pos_wire_actor = self._add_wireframe_overlay(self.pos_mesh)
            if self.neg_mesh:
                self.neg_wire_actor = self._add_wireframe_overlay(self.neg_mesh)
        self._update_title()
        self.plotter.render()

    def set_background(self, color: str):
        self.background_color = color
        self.plotter.set_background(color)
        self._update_title()
        if self.show_atom_labels:
            for a in self.atom_label_actors:
                self.plotter.renderer.RemoveActor(a)
            self.atom_label_actors = []
            self._add_atom_labels()
        self.plotter.render()

    def toggle_wireframe(self, state):
        self.show_wireframe = bool(state)
        if self.pos_wire_actor:
            self.plotter.remove_actor(self.pos_wire_actor); self.pos_wire_actor = None
        if self.neg_wire_actor:
            self.plotter.remove_actor(self.neg_wire_actor); self.neg_wire_actor = None
        if self.show_wireframe and self.pos_mesh and self.neg_mesh:
            self.pos_wire_actor = self._add_wireframe_overlay(self.pos_mesh)
            self.neg_wire_actor = self._add_wireframe_overlay(self.neg_mesh)
        self._update_title(); self.plotter.render()

    def toggle_atom_labels(self, state):
        self.show_atom_labels = bool(state)
        for a in self.atom_label_actors:
            self.plotter.renderer.RemoveActor(a)
        self.atom_label_actors = []
        if self.show_atom_labels:
            self._add_atom_labels()
        self.plotter.render()

    def switch_cube(self, index):
        # Snapshot the full camera state before rebuilding the scene
        cam = self.plotter.renderer.GetActiveCamera()
        saved_pos      = cam.GetPosition()
        saved_focal    = cam.GetFocalPoint()
        saved_viewup   = cam.GetViewUp()
        saved_distance = cam.GetDistance()
        saved_parallel = cam.GetParallelScale()

        # Clear selection state fully before switching
        self.selected_atoms = []
        for a in self.selected_atom_highlights:
            self.plotter.remove_actor(a)
        self.selected_atom_highlights = []
        if self.selection_label:
            self.selection_label.setText("No atoms selected.")
        if self.measurement_label:
            self.measurement_label.setText("")
        if self.selected_atoms_list_widget:
            self.selected_atoms_list_widget.clear()
        self._clear_measurement_actors()

        self.current_cube_index = index
        self.update_visualization()

        # Restore camera exactly — this keeps the same viewpoint regardless of
        # where the new cube's atoms happen to sit in world space
        cam = self.plotter.renderer.GetActiveCamera()
        cam.SetPosition(saved_pos)
        cam.SetFocalPoint(saved_focal)
        cam.SetViewUp(saved_viewup)
        cam.SetDistance(saved_distance)
        cam.SetParallelSca(saved_parallel)
        self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.render()

    # ── Atom picking ──────────────────────────────────────────────────────────

    # ── Geometry measurement helpers ──────────────────────────────────────────

    @staticmethod
    def _bond_length(p1, p2):
        return float(np.linalg.norm(p2 - p1))

    @staticmethod
    def _bond_angle(p1, p2, p3):
        """Angle at p2 in degrees."""
        v1 = p1 - p2; v2 = p3 - p2
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
        return math.degrees(math.acos(float(np.clip(cos_a, -1, 1))))

    @staticmethod
    def _dihedral_angle(p1, p2, p3, p4):
        """Dihedral (torsion) angle p1-p2-p3-p4 in degrees."""
        b1 = p2 - p1; b2 = p3 - p2; b3 = p4 - p3
        n1 = np.cross(b1, b2); n2 = np.cross(b2, b3)
        n1 /= (np.linalg.norm(n1) + 1e-12)
        n2 /= (np.linalg.norm(n2) + 1e-12)
        m1 = np.cross(n1, b2 / (np.linalg.norm(b2) + 1e-12))
        x  = np.dot(n1, n2)
        y  = np.dot(m1, n2)
        return math.degrees(math.atan2(float(y), float(x)))

    # ── Atom picking ──────────────────────────────────────────────────────────

    def _on_left_click(self, interactor, event):
        """Pick nearest atom on left-click and highlight it."""
        interactor.GetInteractorStyle().OnLeftButtonDown()
        x, y     = interactor.GetEventPosition()
        renderer = self.plotter.renderer
        picker   = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        picker.Pick(x, y, 0, renderer)
        if picker.GetCellId() != -1:
            pos = np.array(picker.GetPickPosition())
        else:
          coord = vtk.vtkCoordinate()
          coord.SetCoordinateSystemToDisplay()
          coord.SetValue(float(x), float(y), 0.5)
          pos = np.array(coord.GetComputedWorldValue(renderer))
        cube  = self.cubes[self.current_cube_index]
        dists = np.linalg.norm(cube['coordinates'] - pos, axis=1)
        idx   = int(np.argmin(dists))
        if idx in self.selected_atoms:
            self.selected_atoms.remove(idx)
        else:
            if len(self.selected_atoms) >= self.MAX_SELECTION:
                self.selected_atoms.pop(0)
            self.selected_atoms.append(idx)
        self._update_selected_highlights()
        self._update_selection_label()
        self.plotter.render()

    def atom_pick_callback(self, point):
        pass  # handled by _on_left_click

    def _update_selected_highlights(self):
        for a in self.selected_atom_highlights:
            self.plotter.remove_actor(a)
        self.selected_atom_highlights = []
        cube = self.cubes[self.current_cube_index]
        b2a  = 0.529177 if cube['unit_bohr'] else 1.0
        for i, idx in enumerate(self.selected_atoms):
            e    = self.ELEMENT_DATA.get(cube['atoms'][idx])
            bs_r = (e[3] if e else 0.22) / b2a
            pos  = cube['coordinates'][idx]
            sph  = pv.Sphere(radius=bs_r * 1.45, center=pos,
                             theta_resolution=32, phi_resolution=32)
            col  = ['yellow', 'orange', 'cyan', 'magenta'][i]
            a    = self.plotter.add_mesh(sph, color=col, opacity=0.55,
                                         smooth_shading=True)
            self.selected_atom_highlights.append(a)

    def _update_selection_label(self):
        """Update the plain-text label showing which atoms are selected."""
        if not self.selection_label:
            return
        cube = self.cubes[self.current_cube_index]
        if not self.selected_atoms:
            self.selection_label.setText("No atoms selected.")
            return
        lines = []
        for i, idx in enumerate(self.selected_atoms):
            e   = self.ELEMENT_DATA.get(cube['atoms'][idx], ('?',))
            pos = cube['coordinates'][idx]
            lines.append(f"[{i+1}] {e[0]}{idx+1}  ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        self.selection_label.setText("\n".join(lines))

    # def compute_measurement(self):
    #     """Called by the Measure button."""
    #     cube   = self.cubes[self.current_cube_index]
    #     sel    = self.selected_atoms
    #     n      = len(sel)
    #     coords = [cube['coordinates'][i] for i in sel]

    #     def sym(idx):
    #         e = self.ELEMENT_DATA.get(cube['atoms'][idx], ('?',))
    #         return f"{e[0]}{idx+1}"

    #     if n < 2:
    #         self.measurement_label.setText("Select 2-4 atoms first.")
    #         return
    #     elif n == 2:
    #         d      = self._bond_length(coords[0], coords[1])
    #         d_ang  = d * (0.529177 if cube['unit_bohr'] else 1.0)
    #         result = f"Bond length\n{sym(sel[0])} — {sym(sel[1])}\n= {d_ang:.4f} Ang"
    #     elif == 3:
    #         angle  = self._bond_angle(coords[0], coords[1], coords[2])
    #         result = f"Bond angle\n{sym(sel[0])}-{sym(sel[1])}-{sym(sel[2])}\n= {angle:.3f} deg"
    #     else:
    #         dih    = self._dihedral_angle(coords[0], coords[1], coords[2], coords[3])
    #         result = (f"Dihedral angle\n{sym(sel[0])}-{sym(sel[1])}-"
    #                   f"{sym(sel[2])}-{sym(sel[3])}\n= {dih:.3f} deg")
    #     self.measurement_label.setText(result)
    #     self._draw_measurement_annotation()



    def compute_measurement(self):
        """Called by the Measure button."""
        cube   = self.cubes[self.current_cube_index]
        sel    = self.selected_atoms
        n      = len(sel)
        coords = [cube['coordinates'][i] for i in sel]

        def sym(idx):
            e = self.ELEMENT_DATA.get(cube['atoms'][idx], ('?',))
            return f"{e[0]}{idx+1}"

        if n < 2:
            self.measurement_label.setText("Select 2-4 atoms first.")
            return
        elif n == 2:
            d      = self._bond_length(coords[0], coords[1])
            d_ang  = d * (0.529177 if cube['unit_bohr'] else 1.0)
            result = f"Bond length\n{sym(sel[0])} — {sym(sel[1])}\n= {d_ang:.4f} Ang"
        elif n == 3:
            angle  = self._bond_angle(coords[0], coords[1], coords[2])
            result = f"Bond angle\n{sym(sel[0])}-{sym(sel[1])}-{sym(sel[2])}\n= {angle:.3f} deg"
        else:
            dih    = self._dihedral_angle(coords[0], coords[1], coords[2], coords[3])
            result = (f"Dihedral angle\n{sym(sel[0])}-{sym(sel[1])}-"
                      f"{sym(sel[2])}-{sym(sel[3])}\n= {dih:.3f} deg")
        self.measurement_label.setText(result)
        self._draw_measurement_annotation()

    def clear_selection(self):
        """Clear all selected atoms and reset labels."""
        for a in self.selected_atom_highlights:
            self.plotter.remove_actor(a)
        self.selected_atom_highlights = []
        self.selected_atoms = []
        if self.selection_label:
            self.selection_label.setText("No atoms selected.")
        if self.measurement_label:
            self.measurement_label.setText("")
        self._clear_measurement_actors()
        self.plotter.render()

    # ── S──────────────────────────────────────────────────────────────────

    def toggle_ssao(self, state):
        self.ssao_enabled = bool(state)
        renderer = self.plotter.renderer
        if self.ssao_enabled:
            renderer.SetUseSSAO(True)
            renderer.SetSSAORadius(0.5)
            renderer.SetSSAOBias(0.01)
            renderer.SetSSAOKernelSize(128)
            renderer.SetSSAOBlur(True)
        else:
            renderer.SetUseSSAO(False)
        self.plotter.render()

    # ── Viewport measurement annotations ──────────────────────────────────────

    def _clear_measurement_actors(self):
        for a in self.measurement_actors:
            self.plotter.renderer.RemoveActor(a)
        self.measurement_actors = []

    def _draw_measurement_annotation(self):
        """Draw dashed lines and a floating text label in the 3D viewport."""
        self._clear_measurement_actors()
        cube   = self.cubes[self.current_cube_index]
        sel    = self.selected_atoms
        n      = len(sel)
        if n < 2:
            return
        coords = [cube['coordinates'][i] for i in sel]
        b2a    = 0.529177 if cube['unit_bohr'] else 1.0

        def sym(idx):
            e = self.ELEMENT_DATA.get(cube['atoms'][idx], ('?',))
            return f"{e[0]}{idx+1}"

        ann_color = (1.0, 1.0, 0.2) if self.background_color == 'black' else (0.1, 0.1, 0.7)

        # Draw dashed lines connecting selected atoms
        for k in range(n - 1):
            p1, p2 = np.array(coords[k]), np.array(coords[k+1])
            # Subdivide into segments for dashed appearance
            n_seg = 12
            for s in range(n_seg):
                if s % 2 == 0:
                    t1 = s / n_seg
                    t2 = (s + 1) / n_seg
                    seg = pv.Line(p1 + (p2-p1)*t1, p1 + (p2-p1)*t2)
                    a = self.plotter.add_mesh(seg, color=ann_color,
                                              line_width=2, style='wireframe')
                    self.measurement_actors.append(a)

        # Compute measurement text
        if n == 2:
            d = self._bond_length(coords[0], coords[1]) * (b2a if cube['unit_bohr'] else 1.0)
            text = f"{d:.3f} Ang"
            mid  = (np.array(coords[0]) + np.array(coords[1])) / 2
        elif n == 3:
            angle = self._bond_angle(coords[0], coords[1], coords[2])
            text  = f"{angle:.2f} deg"
            mid   = np.array(coords[1])   # place at vertex atom
        else:
            dih  = self._dihedral_angle(coords[0], coords[1], coords[2], coords[3])
            text = f"{dih:.2f} deg"
            mid  = (np.array(coords[1]) + np.array(coords[2])) / 2

        # Billboard text label at midpoint
        actor = vtk.vtkBillboardTextActor3D()
        actor.SetInput(text)
        actor.SetPosition(*mid)
        actor.SetDisplayOffset(0, 14)
        tp = actor.GetTextProperty()
        tp.SetFontFamilyToArial()
        tp.SetFontSize(14)
        tp.SetBold(True)
        tp.SetColor(*ann_color)
        tp.SetShadow(True)
        tp.SetShadowOffset(1, -1)
        tp.SetJustificationToCentered()
        tp.SetVerticalJustificationToBottom()
        self.plotter.renderer.AddActor(actor)
        self.measurement_actors.append(actor)
        self.plotter.render()

    # ── Animation ─────────────────────────────────────────────────────

    def _anim_step(self):
        """Advance to the next cube, wrapping around."""
        next_idx = (self.current_cube_index + 1) % len(self.cubes)
        self.cube_list.setCurrentRow(next_idx)   # triggers switch_cube via signal

    def toggle_animation(self):
        if len(self.cubes) < 2:
            QMessageBox.information(self.main_window, "Animation",
                                    "Loaeast 2 cube files to animate.")
            return
        if self.anim_playing:
            self.anim_timer.stop()
            self.anim_playing = False
            self.play_btn.setText("▶  Play")
        else:
            if self.anim_timer is None:
                self.anim_timer = QTimer()
                self.anim_timer.timeout.connect(self._anim_step)
            self.anim_timer.start(int(self.anim_spf * 1000))
            self.anim_playing = True
            self.play_btn.setText("⏹  Stop")


    def set_anim_spf(self, slider_value):
        """Slider 1–50 maps to 0.1–5.0 seconds per frame (step 0.1 s)."""
        self.anim_spf = slider_value / 10.0
        if self.spf_label:
            self.spf_label.setText(f"{self.anim_spf:.1f} s / frame")
        if self.anim_playing and self.anim_timer:
            self.anim_timer.setInterval(int(self.anim_spf * 1000))

    # ── Save ────────────────────────────────────────────────────────

    def save_image(self):
        fmt    = self.format_combo.currentText()
        scale  = int(self.resolution_combo.currentText().rstrip('x'))
        transp = (fmt == 'PNG' and
                  self.transp_bg_check is not None and
                  self.transp_bg_check.isChecked())
        default = os.path.splitext(
            os.path.basename(self.cube_files[self.current_cube_index]))[0]
        fname, _ = QFileDialog.getSaveFileName(
            self.main_window, "Save Figure",
            f"{default}.{fmt.lower()}",
            f"{fmt} Files (*.{fmt.lower()})"
        )
        if not fname: return
        if not fname.lower().endswith(f".{fmt.lower()}"):
            fname += f".{fmt.lower()}"

        if self.title_actor:
            self.plotter.renderer.RemoveActor(self.title_actor)
        self.plotter.hide_axes()
        try: self.plotter.remove_scalar_bar()
        except Exception: pass
        renderer  = self.plotter.renderer
        actors_2d = renderer.GetActors2D()
        actors_2d.InitTraversal()
        hidden_2d = []
        actor = actors_2d.GetNextActor2D()
        while actor:
            actor.VisibilityOff(); hidden_2d.append(actor)
            actor = actors_2d.GetNextActor2D()

        w, h = self.plotter.window_size
        renderer.GetRenderWindow().Render()
        raw = self.plotter.screenshot(
            return_img=True,
            window_size=(w * scale, h * scale),
            transparent_background=transp,
        )
        img = Image.fromarray(raw)

        for a in hidden_2d: a.VisibilityOn()
        self._update_title()
        self.plotter.show_axes()
        renderer.GetRenderWindow().Render()

        if fmt == "PNG":
            img.save(fname, format="PNG", dpi=(300, 300))
        elif fmt in ("PDF", "SVG"):
            fig, ax = plt.subplots(
                figsize=(img.width / 300, img.height / 300), dpi=300)
            ax.imshow(img); ax.axis("off")
            fig.savefig(fname, format=fmt.lower(),
                        bbox_inches="tight", pad_inches=0, dpi=300)
            plt.close(fig)

        size_mp = (img.width * img.height) / 1e6
        desc = " (transparent bg)" if transp else ""
        QMessageBox.information(
            self.main_window, "Saved",
            f"Saved: {fname}{desc}\n"
            f"Resolution: {img.width} x {img.height} px  ({size_mp:.1f} MP)"
        )


    # ── Feature 1: Electron density difference map preset ─────────────────────
    def _write_cube(self, cube, fname, comment="Generated by CubeVisualizer"):
        """Write a cube dict back to a Gaussian .cube file."""
        nx, ny, nz = cube['dimensions']
        ox, oy, oz = cube['origin']
        dx, dy, dz = cube['spacing']
        sign = 1 if cube['unit_bohr'] else -1
        with open(fname, 'w') as f:
            f.write(f" {comment}\n Generated\n")
            f.write(f" {len(cube['atoms']):5d}  {ox:12.6f}  {oy:12.6f}  {oz:12.6f}\n")
            f.write(f" {sign*nx:5d}  {dx:12.6f}   0.000000   0.000000\n")
            f.write(f" {sign*ny:5d}   0.000000  {dy:12.6f}   0.000000\n")
            f.write(f" {sign*nz:5d}   0.000000   0.000000  {dz:12.6f}\n")
            for an, pos in zip(cube['atoms'], cube['coordinates']):
                f.write(f" {an:5d}   0.000000  {pos[0]:12.6f}  {pos[1]:12.6f}  {pos[2]:12.6f}\n")
            flat = cube['data'].flatten(order='F')
            for i, v in enumerate(flat):
                f.write(f"  {v:12.5e}")
                if (i + 1) % 6 == 0:
                    f.write("\n")
            if len(flat) % 6 != 0:
                f.write("\n")


    def open_diff_density_dialog(self):
        """Quick preset: ρ(A) − ρ(B) with diverging colormap annotation."""
        if len(self.cubes) < 2:
            QMessageBox.information(self.main_window, "Difference Density",
                "Load at least 2 cube files first.\n"
                "Typically: cube A = excited/product density, cube B = ground/reactant.")
            return
        dlg = QDialog(self.main_window)
        dlg.setWindowTitle("Electron Density Difference Map")
        dlg.setMinimumWidth(380)
        lay = QVBoxLayout(dlg)
        names = [os.path.basename(f) for f in self.cube_files]
        g = QGroupBox("ρ(A) − ρ(B)")
        gl = QGridLayout(g)
        gl.addWidget(QLabel("Density A (e.g. excited state):"), 0, 0)
        ca_combo = QComboBox(); [ca_combo.addItem(n) for n in names]
        gl.addWidget(ca_combo, 0, 1)
        gl.addWidget(QLabel("Density B (e.g. ground state):"), 1, 0)
        cb_combo = QComboBox(); [cb_combo.addItem(n) for n in names]
        if len(names) > 1: cb_combo.setCurrentIndex(1)
        gl.addWidget(cb_combo, 1, 1)
        lay.addWidget(g)
        out_row = QHBoxLayout()
        out_edit = QLineEdit("diff_density.cube")
        out_row.addWidget(QLabel("Output:")); out_row.addWidget(out_edit)
        load_chk = QCheckBox("Load result"); load_chk.setChecked(True)
        out_row.addWidget(load_chk)
        lay.addLayout(out_row)
        info = QLabel(
            "The result will be visualized with the\n"
            "current color scheme: positive lobe =\n"
            "charge gain, negative lobe = charge loss."
        )
        info.setStyleSheet("color: #555; font-size: 9pt;")
        lay.addWidget(info)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("Compute")
        lay.addWidget(btns)
        btns.rejected.connect(dlg.reject)
        def compute():
            ca = self.cubes[ca_combo.currentIndex()]
            cb = self.cubes[cb_combo.currentIndex()]
            if ca['dimensions'] != cb['dimensions']:
                QMessageBox.warning(dlg, "Grid mismatch",
                    "Cubes must have identical grid dimensions.")
                return
            result = dict(
                origin=ca['origin'], spacing=ca['spacing'],
                dimensions=ca['dimensions'],
                atoms=ca['atoms'].copy(), coordinates=ca['coordinates'].copy(),
                data=ca['data'] - cb['data'],
                bonds=ca['bonds'], unit_bohr=ca['unit_bohr'],
            )
            path = out_edit.text().strip() or "diff_density.cube"
            if not path.endswith('.cube'): path += '.cube'
            try:
                self._write_cube(result, path, "Difference density: rho(A)-rho(B)")
            except Exception as e:
                QMessageBox.critical(dlg, "Write error", str(e)); return
            dlg.accept()
            if load_chk.isChecked():
                # Append result to existing list — keep input cubes visible
                idx = len(self.cube_files)
                self.cube_files.append(path)
                self.cubes.append(result)
                if self.cube_list is not None:
                    self.cube_list.blockSignals(True)
                    self.cube_list.addItem(QListWidgetItem(os.path.basename(path)))
                    self.cube_list.blockSignals(False)
                self.clear_selection()
                if self.cube_list is not None:
                    self.cube_list.setCurrentRow(idx)
                else:
                    self.switch_cube(idx)
            else:
                QMessageBox.information(self.main_window, "Done",
                                        f"Saved to:\n{path}")
        btns.accepted.connect(compute)
        dlg.exec_()

    # ── Feature 2: Cube metadata panel ────────────────────────────────────────

    def _cube_metadata_text(self):
        if not self.cubes:
            return "No cube loaded."
        cube = self.cubes[self.current_cube_index]
        nx, ny, nz = cube['dimensions']
        dx, dy, dz = cube['spacing']
        unit = "Bohr" if cube['unit_bohr'] else "Å"
        b2a  = 0.529177 if cube['unit_bohr'] else 1.0
        dx_a, dy_a, dz_a = dx*b2a, dy*b2a, dz*b2a
        vol_voxel = dx_a * dy_a * dz_a
        d = cube['data']
        dmin, dmax, dmean = d.min(), d.max(), d.mean()
        iso = self.current_isovalue
        pct = 00.0 * iso / dmax if dmax > 0 else 0.0
        # Volume enclosed estimate (voxels above isovalue × voxel volume)
        vol_pos = float(np.sum(d >= iso) * vol_voxel)
        vol_neg = float(np.sum(d <= -iso) * vol_voxel)
        fname   = os.path.basename(self.cube_files[self.current_cube_index])
        return (
            f"File:       {fname}\n"
            f"Grid:       {nx} × {ny} × {nz}  ({nx*ny*nz:,} pts)\n"
            f"Voxel:      {dx_a:.4f} × {dy_a:.4f} × {dz_a:.4f} Å\n"
            f"Uni   {unit}\n"
            f"Atoms:      {len(cube['atoms'])}\n"
            f"Data min:   {dmin:.5f}\n"
            f"Data max:   {dmax:.5f}\n"
            f"Data mean:  {dmean:.5f}\n"
            f"Isovalue:   {iso:.4f}  ({pct:.1f}% of max)\n"
            f"Vol (+iso): {vol_pos:.3f} Å³\n"
            f"Vol (−iso): {vol_neg:.3f} Å³"
        )

    def refresh_metadata(self):
        if self.metadata_label:
            self.metadata_label.setText(self._cube_metadata_text())

    # ── Feature 3: Isostistics ──────────────────────────────────────

    def compute_isosurface_stats(self):
        """Compute enclosed volume and integrated electron count."""
        if not self.cubes:
            return
        cube = self.cubes[self.current_cube_index]
        d    = cube['data']
        b2a  = 0.529177 if cube['unit_bohr'] else 1.0
        dx, dy, dz = [s * b2a for s in cube['spacing']]
        vol_voxel  = dx * dy * dz          #l
        iso = self.current_isovalue

        n_pos  = int(np.sum(d >=  iso))
        n_neg  = int(np.sum(d <= -iso))
        vol_p  = n_pos * vol_voxel
        vol_n  = n_neg * vol_voxel
        # Electron count: integrate |ψ|² (or ρ) over enclosed voxels
        elec_p = float(np.sum(d[d >=  iso]) * vol_voxel)
        elec_n = float(np.sum(np.abs(d[d <= -iso])) * vol_voxel)

        msg = (
            f"Isovalue: {iso:.4f}\n\n"
            f"Positive lobe\n"
            f"  Voxels:   {n_pos:,}\n"
         f"  Volume:   {vol_p:.3f} Å³\n"
            f"  ∫ρ dV :   {elec_p:.4f} e\n\n"
            f"Negative lobe\n"
            f"  Voxels:   {n_neg:,}\n"
            f"  Volume:   {vol_n:.3f} Å³\n"
            f"  ∫|ρ| dV:  {elec_n:.4f} e"
        )
        QMessageBox.information(self.main_window, "Isosurface Statistics", msg)

    # ── Feature 4: Camera presets ──────────────────────────────────────────── def set_camera_preset(self, preset):
        views = {
            'Front':       dict(position=(0,-10, 0), up=(0, 0, 1)),
            'Back':        dict(position=(0, 10, 0), up=(0, 0, 1)),
            'Top':         dict(position=(0, 0, 10), up=(0, 1, 0)),
            'Bottom':      dict(position=(0, 0,-10), up=(0, 1, 0)),
            'Left':        dict(position=(-10, 0, 0), up=(0, 0, 1)),
            'Right':       dict(position=( 10, 0, 0), up=(0, 0, 1)),
            'Perspective': dict(position=(6, -8, 5), up=(0, 0, 1)),
        }
        if preset not in views or not self.cubes:
            return
        # Find centroid of molecule
        cube     = self.cubes[self.current_cube_index]
        centroid = cube['coordinates'].mean(axis=0) if len(cube['coordinates']) else np.zeros(3)
        v        = views[preset]
        pos_dir  = np.array(v['position'], dtype=float)
        # Scale to a comfortable distance
        extents  = cube['coordinates'].ptp(axis=0) if len(cube['coordinates']) else np.ones(3)*5
        dist     = max(extents.max() * 2.5, 8.0)
        pos_dir  = pos_dir / (np.linalg.norm(pos_dir) + 1e-9) * dist
        cam = self.plotter.renderer.GetActiveCamera()
        cam.SetFocalPoint(*centroid)
        cam.SetPosition(*(centroid + pos_dir))
        cam.SetViewUp(*v['up'])
        self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.render()

    # ── Feature 5: Export all cubes as image sequence ────────────────────────   def export_image_sequence(self):
        if len(self.cubes) < 1:
            QMessageBox.information(self.main_window, "Export Sequence",
                                    "Load at least one cube file first.")
            return
        dlg = QDialog(self.main_window)
        dlg.setWindowTitle("Export Image Sequence")
        dlg.setMinimumWidth(380)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("Output folder:"))
        folder_row = QHBoxLayout()
        folder_edit = QLineEdit(os.path.expanduser("~"))
        folder_btn  = QPushButton("Browse…")
        def browse_folder():
            d = QFileDialog.getExistingDirectory(dlg, "Select output folder")
            if d: folder_edit.setText(d)
        folder_btn.clicked.connect(browse_folder)
        folder_row.addWidget(folder_edit); folder_row.addWidget(folder_btn)
        lay.addLayout(folder_row)
        prefix_row = QHBoxLayout()
        prefix_row.addWidget(QLabel("Filename prefix:"))
        prefix_edit = QLineEdit("frame")
        prefix_row.addWidget(prefix_edit)
        lay.addLayout(prefix_row)
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Resolution:"))
        scale_combo = QComboBox()
        scale_combo.addItems(["1x", "2x", "3x"])
        scale_combo.setCurrentIndex(1)
        scale_row.addWidget(scale_combo)
        lay.addLayout(scale_row)
        transp_seq_check = QCheckBox("Transparent background")
        transp_seq_check.setChecked(False)
        transp_seq_check.setToolTip("Export PNGs with alpha channel — no background colour")
        lay.addWidget(transp_seq_check)
        lay.addWidget(QLabel(
            f"Will export {len(self.cubes)} PNG(s) at current\n"
            "viewpoint, isovalue and rendering settings."))
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("Export")
        lay.addWidget(btns)
        btns.rejected.connect(dlg.reject)
        def do_export():
            folder = folder_edit.text()
            prefix = prefix_edit.text() or "frame"
            scale  = int(scale_combo.currentText().rstrip('x'))
            transp = transp_seq_check.isChecked()
            if not os.path.isdir(folder):
                QMessageBox.warning(dlg, "Error", "Folder does not exist."); return
            dlg.accept()
            orig_idx = self.current_cube_index
            # Save camera
            cam = self.plotter.renderer.GetActiveCamera()
            sv_pos = cam.GetPosition(); sv_foc = cam.GetFocalPoint()
            sv_up  = cam.GetViewUp()
            saved  = []
            for i in range(len(self.cubes)):
                self.switch_cube(i)
                # Restore camera so view is consistent
                cam.SetPosition(sv_pos); cam.SetFocalPoint(sv_foc)
                cam.SetViewUp(sv_up)
                self.plotter.renderer.ResetCameraClippingRange()
                self.plotter.render()
                # Hide UI overlays
                if self.title_actor:
                    self.plotter.renderer.RemoveActor(self.title_actor)
                self.plotter.hide_axes()
                renderer  = self.plotter.renderer
                actors_2d = renderer.GetActors2D()
                actors_2d.InitTraversal(); hidden = []
                a = actors_2d.GetNextActor2D()
                while a: a.VisibilityOff(); hidden.append(a); a = actors_2d.GetNextActor2D()
                renderer.GetRenderWindow().Render()
                w, h = self.plotter.window_size
                raw  = self.plotter.screenshot(return_img=True,
                                               window_size=(w*scale, h*scale),
                                               transparent_background=transp)
                for a in hidden: a.VisibilityOn()
                self._update_title(); self.plotter.show_axes()
                renderer.GetRenderWindow().Render()
                out = os.path.join(folder, f"{prefix}_{i+1:04d}.png")
                Image.fromarray(raw).save(out, format="PNG", dpi=(300, 300))
                saved.append(out)
            self.switch_cube(orig_idx)
            QMessageBox.information(self.main_window, "Done",
                f"Exported {len(saved)} image(s) to:\n{folder}")
        btns.accepted.connect(do_export)
        dlg.exec_()

    # ── Feature 9: Session save / restore ────────────────────────────────────

    def open_cube_files_dialog(self):
        """Open a file dialog to select one or more .cube files (replaces current list)."""
        fnames, _ = QFileDialog.getOpenFileNames(
            self.main_window, "Open Cube Files", "",
            "Cube Files (*.cube);;All Files (*)")
        if not fnames:
            return
        # Clear old data
        self.cube_files = []; self.cubes = []; self.current_cube_index = 0
        if self.cube_list is not None:
            self.cube_list.blockSignals(True); self.cube_list.clear()
            self.cube_list.blockSignals(False)
        self.clear_selection()

        first_new_idx = None
        for fname in fnames:
            fname = os.path.abspath(fname)
            try:
                cube = self.read_cube(fname)
                idx  = len(self.cube_files)
                self.cube_files.append(fname); self.cubes.append(cube)
                if self.cube_list is not None:
                    self.cube_list.blockSignals(True)
                    self.cube_list.addItem(QListWidgetItem(os.path.basename(fname)))
                    self.cube_list.blockSignals(False)
                if first_new_idx is None:
                    first_new_idx = idx
            except Exception as e:
                QMessageBox.warning(self.main_window, "Load Error",
                                    f"Could not load:\n{fname}\n\n{e}")
        if first_new_idx is not None:
            if self.cube_list is not None:
                self.cube_list.setCurrentRow(first_new_idx)
            else:
                self.switch_cube(first_new_idx)
 

    def open_cube_operations_dialog(self):
        """Show the cube operations dialog."""
        if len(self.cubes) < 1:
            QMessageBox.information(self.main_window, "Cube Operations",
                                    "Load at least one cube file first.")
            return
        dlg = QDialog(self.main_window)
        dlg.setWindowTitle("Cube Operations"); dlg.setMinimumWidth(420)
        layout = QVBoxLayout(dlg)
        names = [os.path.basename(f) for f in self.cube_files]
        op_box = QGroupBox("Operation"); op_grid = QGridLayout(op_box)
        op_combo = QComboBox()
        op_combo.addItems(["A + B", "A \u2212 B", "A \u00d7 B", "A \u00f7 B", "A \u00d7 scalar"])
        op_grid.addWidget(QLabel("Operation:"), 0, 0); op_grid.addWidget(op_combo, 0, 1, 1, 2)
        op_grid.addWidget(QLabel("Cube A:"), 1, 0)
        a_combo = QComboBox(); [a_combo.addItem(n) for n in names]
        op_grid.addWidget(a_combo, 1, 1, 1, 2)
        b_label = QLabel("Cube B:"); b_combo = QComboBox()
        [b_combo.addItem(n) for n in names]
        if len(names) > 1: b_combo.setCurrentIndex(1)
        op_grid.addWidget(b_label, 2, 0); op_grid.addWidget(b_combo, 2, 1, 1, 2)
        scalar_label = QLabel("Scalar value:")
        scalar_spin = QDoubleSpinBox(); scalar_spin.setRange(-1e9,1e9)
        scalar_spin.setDecimals(6); scalar_spin.setValue(1.0)
        scalar_spin.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)
        op_grid.addWidget(scalar_label, 3, 0); op_grid.addWidget(scalar_spin, 3, 1)
        def on_op_changed(text):
            is_scalar = "scalar" in text
            b_label.setVisible(not is_scalar); b_combo.setVisible(not is_scalar)
            scalar_label.setVisible(is_scalar); scalar_spin.setVisible(is_scalar)
        op_combo.currentTextChanged.connect(on_op_changed); on_op_changed(op_combo.currentText())
        layout.addWidget(op_box)
        out_box = QGroupBox("Output"); out_layout = QHBoxLayout(out_box)
        out_edit = QLineEdit("result.cube")
        out_btn = QPushButton("Browse…")
        def browse_out():
            path, _ = QFileDialog.getSaveFileName(dlg, "Save result cube",
                                                  out_edit.text(), "Cube Files (*.cube)")
            if path: out_edit.setText(path)
        out_btn.clicked.connect(browse_out)
        load_check = QCheckBox("Load result into viewer"); load_check.setChecked(True)
        out_layout.addWidget(out_edit); out_layout.addWidget(out_btn)
        out_layout.addWidget(load_check); layout.addWidget(out_box)
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.button(QDialogButtonBox.Ok).setText("Compute")
        layout.addWidget(btn_box); btn_box.rejected.connect(dlg.reject)
        def compute():
            op = op_combo.currentText(); ca = self.cubes[a_combo.currentIndex()]
            if "scalar" in op:
                result_data = ca['data'] * scalar_spin.value(); base_cube = ca
            else:
                cb = self.cubes[b_combo.currentIndex()]
                if ca['dimensions'] != cb['dimensions']:
                    QMessageBox.warning(dlg, "Grid mismatch",
                        f"Cubes have different grid dimensions:\n"
                        f"  A: {ca['dimensions']}\n  B: {cb['dimensions']}"); return
                if   op == "A + B":             result_data = ca['data'] + cb['data']
                elif op == "A \u2212 B":         result_data = ca['data'] - cb['data']
                elif op == "A \u00d7 B":         result_data = ca['data'] * cb['data']
                else:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        result_data = np.where(np.abs(cb['data']) > 1e-30,
                                               ca['data'] / cb['data'], 0.0)
                base_cube = ca
            result_cube = dict(origin=base_cube['origin'], spacing=base_cube['spacing'],
                               dimensions=base_cube['dimensions'],
                               atoms=base_cube['atoms'].copy(),
                               coordinates=base_cube['coordinates'].copy(),
                               data=result_data, bonds=base_cube['bonds'],
                               unit_bohr=base_cube['unit_bohr'])
            out_path = out_edit.text().strip() or "result.cube"
            if not out_path.lower().endswith('.cube'): out_path += '.cube'
            try:
                self._write_cube(result_cube, out_path, comment=f"CubeVisualizer: {op}")
            except Exception as e:
                QMessageBox.critical(dlg, "Write error", str(e)); return
            dlg.accept()
            if load_check.isChecked():
                idx = len(self.cube_files)
                self.cube_files.append(out_path); self.cubes.append(result_cube)
                if self.cube_list is not None:
                    self.cube_list.blockSignals(True)
                    self.cube_list.addItem(QListWidgetItem(os.path.basename(out_path)))
                    self.cube_list.blockSignals(False)
                self.clear_selection()
                if self.cube_list is not None:
                    self.cube_list.setCurrentRow(idx)
                else:
                    self.switch_cube(idx)
            else:
                QMessageBox.information(self.main_window, "Done",
                                        f"Result saved to:\n{out_path}")
        btn_box.accepted.connect(compute); dlg.exec_()


    def save_session(self):
        fname, _ = QFileDialog.getSaveFileName(
            self.main_window, "Save Session",
            "session.json", "Session Files (*.json)")
        if not fname: return
        cam = self.plotter.renderer.GetActiveCamera()
        state = {
            'cube_files':      self.cube_files,
            'current_index':   self.current_cube_index,
            'isovalue':        self.current_isovalue,
            'opacity':         self.surface_opacity,
            'mol_theme':       self.current_mol_theme,
            'background':      self.background_color,
            'show_wireframe':  self.show_wireframe,
            'show_labels':     self.show_atom_labels,
            'ssao':            self.ssao_enabled,
            'color_scheme':    self.color_scheme_combo.currentText()
                               if self.color_scheme_combo else 'Mathematica (default)',
            'camera': {
                'position':  list(cam.GetPosition()),
                'focal':     list(cam.GetFocalPoint()),
                'viewup':    list(cam.GetViewUp()),
                'distance':  cam.GetDistance(),
            },
        }
        try:
            with open(fname, 'w') as f:
                json.dump(state, f, indent=2)
            QMessageBox.information(self.main_window, "Session Saved",
                                    f"Session saved to:\n{fname}")
        except Exception as e:
            QMessageBox.critical(self.main_window, "Save Error", str(e))

    def load_session(self):
        fname, _ = QFileDialog.getOpenFileName(
            self.main_window, "Load Session",
            "", "Session Files (*.json)")
        if not fname: return
        try:
            with open(fname) as f:
                state = json.load(f)
        except Exception as e:
            QMessageBox.critical(self.main_window, "Load Error", str(e)); return

        # Reload cube files
        valid = [f for f in state.get('cube_files', []) if os.path.isfile(f)]
        missing = [f for f in state.get('cube_files', []) if not os.path.isfile(f)]
        if missing:
            QMessageBox.warning(self.main_window, "Missing Files",
                "Some cube files could not be found:\n" +
                "\n".join(os.path.basename(m) for m in missing))
        if not valid:
            QMessageBox.critical(self.main_window, "No Files",
                                 "No valid cube files found for this session.")
            return

        self.cube_files = []; self.cubes = []; self.current_cube_index = 0
        if self.cube_list is not None:
            self.cube_list.blockSignals(True); self.cube_list.clear()
        for f in valid:
            try:
                self.cubes.append(self.read_cube(f))
                self.cube_files.append(f)
                if self.cube_list is not None:
                    self.cube_list.addItem(QListWidgetItem(os.path.basename(f)))
            except Exception: pass
        if self.cube_list is not None:
            self.cube_list.blockSignals(False)

        # Restore scalar settings
        self.current_isovalue  = state.get('isovalue', 0.03)
        self.surface_opacity   = state.get('opacity', 1.0)
        self.current_mol_theme = state.get('mol_theme', 'Ball and Stick')
        self.background_color  = state.get('background', 'black')
        self.show_wireframe    = state.get('show_wireframe', True)
        self.show_atom_labels  = state.get('show_labels', False)
        self.ssao_enabled      = state.get('ssao', False)

        # Restore color scheme
        scheme = state.get('color_scheme', 'Mathematica (default)')
        if scheme in LOBE_COLOR_SCHEMES:
            self.lobe_pos_color, self.lobe_neg_color = LOBE_COLOR_SCHEMES[scheme]
            if self.color_scheme_combo:
                self.color_scheme_combo.blockSignals(True)
                self.color_scheme_combo.setCurrentText(scheme)
                self.color_scheme_combo.blockSignals(False)

        # Rebuild scene
        idx = min(state.get('current_index', 0), len(self.cubes) - 1)
        self.plotter.set_background(self.background_color)
        self.switch_cube(idx)
        if self.cube_list is not None:
            self.cube_list.setCurrentRow(idx)

        # Restore camera
        cam_s = state.get('camera', {})
        if cam_s:
            cam = self.plotter.renderer.GetActiveCamera()
            cam.SetPosition(*cam_s['position'])
            cam.SetFocalPoint(*cam_s['focal'])
            cam.SetViewUp(*cam_s['viewup'])
            cam.SetDistance(cam_s['distance'])
            self.plotter.renderer.ResetCameraClippingRange()
            self.plotter.render()

        QMessageBox.information(self.main_window, "Session Loaded",
                                f"Session restored from:\n{fname}")

    # ═══════════════════════════════════════════════════════════════════════════
    # UI – Toolbar + dropdown panels + side panel
    # ══════════════════════════════════════════════════════════════════âdef _separator(self):
        line = QFrame(); line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken); return line

    # ── Dropdown panel content builders ──────────────────────────────────────

    def _build_files_content(self, parent):
        lay = QVBoxLayout(parent); lay.setContentsMargins(8,6,8,8)
        open_btn  = QPushButton("📂  Open Cube Files…")
        open_btn.clicked.connect(self.open_cube_files_dialog)
        ses_save  = QPushButton("💾  Save Session…")
        ses_save.clicked.connect(self.save_session)
        ses_load  = QPushButton("📖  Load Session…")
        ses_load.clicked.connect(self.load_session)
        quit_btn  = QPushButton("✖  Close Program")
        quit_btn.clicked.connect(self.main_window.close)
        for b in [open_btn, ses_save, ses_load, self._separator(), quit_btn]:
            if isinstance(b, QFrame): lay.addWidget(b)
            else:
                b.setMinimumHeight(32)
                b.setStyleSheet("text-align:left; padding-left:8px;")
                lay.addWidget(b)
                
    def _build_appearance_content(self, parent):
        lay = QVBoxLayout(parent); lay.setContentsMargins(8,6,8,8)
        # Molecule theme
        lay.addWidget(QLabel("<b>Molecular Representation</b>"))
        self.mol_btn_group = QButtonGroup()
        for tn in self.MOLECULE_THEMES:
            rb = QRadioButton(tn); rb.setChecked(tn == self.current_mol_theme)
            rb.toggled.connect(lambda chk, t=tn: chk and self.set_molecule_theme(t))
            self.mol_btn_group.addButton(rb); lay.addWidget(rb)
        lay.addWidget(self._separator())
        lay.addWidget(QLabel("<b>Isosurface Colors</b>"))
        self.color_scheme_combo = QComboBox()
        for n in LOBE_COLOR_SCHEMES: self.color_scheme_combo.addItem(n)
        self.color_scheme_combo.currentTextChanged.connect(self.set_lobe_colors)
        lay.addWidget(self.color_scheme_combo)
        lay.addWidget(self._separator())
        lay.addWidget(QLabel("<b>Opacity</b>"))
        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Low"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(10); self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(int(self.surface_opacity*100))
        self.opacity_slider.valueChanged.connect(self.update_opacity)
        op_row.addWidget(self.opacity_slider)
        op_row.addWidget(QLabel("High"))
        lay.addLayout(op_row)
        self.opacity_label_value = QLabel(f"{self.surface_opacity:.2f}")
        self.opacity_label_value.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.opacity_label_value)
        lay.addWidget(self._separator())
        lay.addWidget(QLabel("<b>Background & Lighting</b>"))
        self.bg_btn_group = QButtonGroup()
        for lbl, col in [("Black","black"),("White","white")]:
            rb = QRadioButton(lbl); rb.setChecked(col==self.background_color)
            rb.toggled.connect(lambda chk,c=col: chk and self.set_background(c))
            self.bg_btn_group.addButton(rb); lay.addWidget(rb)
        ssao = QCheckBox("Ambient Occlusion (SSAO)")
        ssao.setChecked(self.ssao_enabled); ssao.stateChanged.connect(self.toggle_ssao)
        lay.addWidget(ssao)
        lay.addWidget(self._separator())
        wf = QCheckBox("Mesh Overlay on Lobes")
        wf.setChecked(self.show_wireframe); wf.stateChanged.connect(self.toggle_wireframe)
        lay.addWidget(wf)
        lbl = QCheckBox("Show Atom Labels")
        lbl.setChecked(self.show_atom_labels); lbl.stateChanged.connect(self.toggle_atom_labels)
        lay.addWidget(lbl)

    def _build_camera_content(self, parent):
        lay = QVBoxLayout(parent); lay.setContentsMargins(8,6,8,8)
        lay.addWidget(QLabel("<b>Standard Views</b>"))
        grid = QGridLayout()
        presets = ['Front','Back','Top','Bottom','Left','Right','Perspective']
        for i, p in enumerate(presets):
            btn = QPushButton(p); btn.setFixedHeight(30)
            btn.clicked.connect(lambda _, pr=p: self.set_camera_preset(pr))
            grid.addWidget(btn, i//4, i%4)
        lay.addLayout(grid)

    def _build_measure_content(self, parent):
        lay = QVBoxLayout(parent); lay.setContentsMargins(8,6,8,8)
        lay.addWidget(QLabel("<b>Click atoms in the viewport to select:</b>"))
        self.selection_label = QLabel("No atoms selected.")
        self.selection_label.setWordWrap(True)
        self.selection_label.setStyleSheet(
            "background:#fff;color:#111;border:1px solid #aaa;padding:4px;min-height:48px;")
        lay.addWidget(self.selection_label)
        br = QHBoxLayout()
        mb = QPushButton("Measure"); mb.clicked.connect(self.compute_measurement)
        cb = QPushButton("Clear");   cb.clicked.connect(self.clear_selection)
        br.addWidget(mb); br.addWidget(cb); lay.addLayout(br)
        self.measurement_label = QLabel("")
        self.measurement_label.setWordWrap(True)
        self.measurement_label.setStyleSheet(
            "background:#eef6ff;color:#00008B;border:1px solid #aac;"
            "padding:6px;font-weight:bold;min-height:44px;")
        lay.addWidget(self.measurement_label)

    def _build_analysis_content(self, parent):
        lay = QVBoxLayout(parent); lay.setContentsMargins(8,6,8,8)
        for lbl, tip, fn in [
            ("Isosurface Statistics",    "Volume & electron count", self.compute_isosurface_stats),
            ("Difference Density Map…",  "ρ(A)−ρ(B)",              self.open_diff_density_dialog),
            ("Cube Operations…",         "Add/subtract/scale cubes",self.open_cube_operations_dialog),
        ]:
            btn = QPushButton(lbl); btn.setToolTip(tip)
            btn.setMinimumHeight(32)
            btn.setStyleSheet("text-align:left;padding-left:8px;")
            btn.clicked.connect(fn); lay.addWidget(btn)


    def _build_export_content(self, parent):
        lay = QVBoxLayout(parent); lay.setContentsMargins(8,6,8,8)
        lay.addWidget(QLabel("<b>Save Current View</b>"))
        fr = QHBoxLayout()
        fr.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox(); self.format_combo.addItems(["PNG","PDF","SVG"])
        fr.addWidget(self.format_combo)
        fr.addWidget(QLabel("Res:"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["1x","2x","3x","4x"]); self.resolution_combo.setCurrentIndex(1)
        fr.addWidget(self.resolution_combo); lay.addLayout(fr)
        self.transp_bg_check = QCheckBox("Transparent background (PNG only)")
        lay.addWidget(self.transp_bg_check)
        sb = QPushButton("Save Figure…"); sb.clicked.connect(self.save_image); lay.addWidget(sb)
        lay.addWidget(self._separator())
        lay.addWidget(QLabel("<b>Image Sequence</b>"))
        seq = QPushButton("Export All Cubes as PNG Sequence…")
        seq.clicked.connect(self.export_image_sequence); lay.addWidget(seq)

    # ── Dropdown panel system ─────────────────────────────────────────────────

    def _toggle_dropdown(self, key, builder_fn, anchor_btn):
        """Show or hide the dropdown panel for the given key under anchor_btn."""
        # Close any other open dropdown first
        for k, info in list(self._dropdowns.items()):
            if k != key and info['panel'].isVisible():
                info['panel'].hide()
                info['btn'].setChecked(False)

        panel = self._dropdowns[key]['panel']
        if panel.isVisible():
            panel.hide()
            anchor_btn.setChecked(False)
        else:
            self._position_dropdown(panel, anchor_btn)
            panel.show(); panel.raise_()
            anchor_btn.setChecked(True)

    def _position_dropdown(self, panel, anchor_btn):
        """Position the dropdown panel directly below the toolbar button."""
        # Map button bottom-left to global screen coords, then to main window coords
        btn_global = anchor_btn.mapToGlobal(anchor_btn.rect().bottomLeft())
        local = self.main_window.mapFromGlobal(btn_global)
        panel.move(local)
        # Constrain so it doesn't go off-screen horizontally
        mw = self.main_window.width()
        if local.x() + panel.width() > mw:
            panel.move(mw - panel.width(), local.y())

    def _close_all_dropdowns(self):
        for info in self._dropdowns.values():
            info['panel'].hide()
            info['btn'].setChecked(False)

    # ── Side panel (always visible) ───────────────────────────────────────────

    def _build_side_panel(self):
        """Permanent left-side panel: cube list + isovalue slider."""
        w = QWidget(); w.setFixedWidth(220)
        w.setStyleSheet("QWidget { background: #1e1e2e; color: #cdd6f4; }")
        lay = QVBoxLayout(w); lay.setContentsMargins(6,8,6,8); lay.setSpacing(6)

        title = QLabel("CUBES")
        title.setStyleSheet("font-size:9pt;font-weight:bold;color:#89b4fa;letter-spacing:2px;")
        title.setAlignment(Qt.AlignCenter)
        lay.addWidget(title)

        self.cube_list = QListWidget()
        self.cube_list.setStyleSheet(
            "QListWidget { background:#181825; border:1px solid #313244; color:#cdd6f4; font-size:10pt; }"
            "QListWidget::item { padding:5px 4px; border-bottom:1px solid #313244; }"
            "QListWidget::item:selected { background:#89b4fa; color:#1e1e2e; font-weight:bold; }"
            "QListWidget::item:hover { background:#313244; }")
        self.cube_list.setMinimumHeight(180)
        for fname in self.cube_files:
            self.cube_list.addItem(QListWidgetItem(os.path.basename(fname)))
        if self.cube_files:
            self.cube_list.setCurrentRow(self.current_cube_index)
        self.cube_list.currentRowChanged.connect(self.switch_cube)
        lay.addWidget(self.cube_list)

        # Animation strip
        anim_frame = QFrame()
        anim_frame.setStyleSheet("QFrame{background:#181825;border:1px solid #313244;border-radius:4px;}")
        af = QVBoxLayout(anim_frame); af.setContentsMargins(6,4,6,4); af.setSpacing(3)
        self.play_btn = QPushButton("▶  Play")
        self.play_btn.setStyleSheet(
            "QPushButton{background:#313244;color:#cdd6f4;border-radius:4px;padding:4px;}"
            "QPushButton:hover{background:#45475a;}"
            "QPushButton:checked{background:#89b4fa;color:#1e1e2e;}")
        self.play_btn.setCheckable(True)
        self.play_btn.clicked.connect(self.toggle_animation)
        af.addWidget(self.play_btn)
        spf_row = QHBoxLayout()
        spf_row.addWidget(QLabel("Fast"))
        self.spf_slider = QSlider(Qt.Horizontal)
        self.spf_slider.setMinimum(1); self.spf_slider.setMaximum(50)
        self.spf_slider.setValue(int(self.anim_spf*10))
        self.spf_slider.valueChanged.connect(self.set_anim_spf)
        spf_row.addWidget(self.spf_slider)
        spf_row.addWidget(QLabel("Slow"))
        af.addLayout(spf_row)
        self.spf_label = QLabel(f"{self.anim_spf:.1f} s/fr")
        self.spf_label.setAlignment(Qt.AlignCenter)
        self.spf_label.setStyleSheet("font-size:8pt;color:#a6adc8;")
        af.addWidget(self.spf_label)
        lay.addWidget(anim_frame)

        # Separator + Isovalue
        sep = QLabel("ISOVALUE")
        sep.setStyleSheet("font-size:9pt;font-weight:bold;color:#89b4fa;letter-spacing:2px;margin-top:4px;")
        sep.setAlignment(Qt.AlignCenter)
        lay.addWidget(sep)

        iso_frame = QFrame()
        iso_frame.setStyleSheet("QFrame{background:#181825;border:1px solid #313244;border-radius:4px;}")
        iso_lay = QVBoxLayout(iso_frame); iso_lay.setContentsMargins(6,6,6,6); iso_lay.setSpacing(4)
        max_val = max((c['data'].max() for c in self.cubes), default=0.1)
        max_val = max(max_val, 0.001)
        self.isovalue_slider = QSlider(Qt.Horizontal)
        self.isovalue_slider.setMinimum(1)
        self.isovalue_slider.setMaximum(int(max_val*10000))
        self.isovalue_slider.setValue(int(self.current_isovalue*10000))
        self.isovalue_slider.valueChanged.connect(lambda v: self.update_isovalue(v/10000.0))
        iso_lay.addWidget(self.isovalue_slider)
        iso_edit_row = QHBoxLayout()
        self.isovalue_edit = QLineEdit(f"{self.current_isovalue:.4f}")
        self.isovalue_edit.setStyleSheet("background:#313244;color:#cdd6f4;border:1px solid #45475a;border-radius:3px;padding:2px;")
        self.isovalue_edit.setValidator(QDoubleValidator(0.0001, max_val, 4))
        self.isovalue_edit.returnPressed.connect(lambda: self.update_isovalue(self.isovalue_edit.text()))
        iso_edit_row.addWidget(self.isovalue_edit)
        iso_lay.addLayout(iso_edit_row)
        # Metadata
        self.metadata_label = QLabel(self._cube_metadata_text())
        self.metadata_label.setWordWrap(True)
        self.metadata_label.setStyleSheet("font-family:monospace;font-size:7pt;color:#a6adc8;margin-top:4px;")
        iso_lay.addWidget(self.metadata_label)
        lay.addWidget(iso_frame)

        lay.addStretch()
        return w

    # ── Toolbar builder ───────────────────────────────────────────────────────

    def create_toolbar_and_side_panel(self):
        """Top toolbar with dropdown panels + persistent left side panel."""
        self._dropdowns = {}

        # ── Top toolbar ───────────────────────────────────────────────────
        tb = QToolBar("Main", self.main_window)
        tb.setMovable(False); tb.setFloatable(False)
        tb.setFixedHeight(48)
        tb.setStyleSheet(
            "QToolBar { background:#1e1e2e; border-bottom:2px solid #313244; spacing:2px; padding:2px; }"
        )
        self.main_window.addToolBar(Qt.TopToolBarArea, tb)

        MENUS = [
            ("📂  Filess",    "files",  self._build_files_content,      200),
            ("🎨  Appearance", "appearance", self._build_appearance_content, 280),
            ("📷  Camera",     "camera",     self._build_camera_content,     260),
            ("📐  Measure",    "measure",    self._build_measure_content,    320),
            ("🔬  Analysis",   "analysis",   self._build_analysis_content,   240),
            ("💾  Export",     "export",     self._build_export_content,     280),
        ]


        BTN_STYLE = (
            "QPushButton {"
            "  color:#cdd6f4; background:transparent; border:none;"
            "  font-size:11pt; padding:4px 14px; border-radius:6px;"
            "}"
            "QPushButton:hover   { background:#313244; }"
            "QPushButton:checked { background:#89b4fa; color:#1e1e2e; font-weight:bold; }"
        )

        PANEL_STYLE = (
            "QWidget#dropdown {"
            "  background:#2a2a3e; border:1px solid #45475a;"
            "  border-top:none; border-radius:0 0 8px 8px;"
            "}"
            "QLabel { color:#cdd6f4; font-size:10pt; }"
            "QPushButton { color:#cdd6f4; background:#313244; border:1px solid #45475a;"
            "  border-radius:4px; padding:5px 8px; }"
            "QPushButton:hover { background:#45475a; }"
            "QComboBox { background:#313244; color:#cdd6f4; border:1px solid #45475a; padding:3px; }"
            "QCheckBox { color:#cdd6f4; }"
            "QRadioButton { color:#cdd6f4; }"
            "QSlider::groove:horizontal { background:#313244; height:4px; border-radius:2px; }"
            "QSlider::handle:horizontal { background:#89b4fa; width:12px; height:12px;"
            "  margin:-4px 0; border-radius:6px; }"
            "QLineEdit { background:#313244; color:#cdd6f4; border:1px solid #45475a; border-radius:3px; padding:2px; }"
        )

        for label, key, builder_fn, width in MENUS:
            btn = QPushButton(label, self.main_window)
            btn.setCheckable(True)
            btn.setStyleSheet(BTN_STYLE)
            btn.setFixedHeight(40)
            btn.setMinimumWidth(110)

            # Build the dropdown panel as a child of main_window (not toolbar)
            # so it can overlap the viewport
            panel = QWidget(self.main_window)
            panel.setObjectName("dropdown")
            panel.setFixedWidth(width)
            panel.setStyleSheet(PANEL_STYLE)
            builder_fn(panel)
            panel.adjustSize()
            panel.hide()

            self._dropdowns[key] = {'btn': btn, 'panel': panel}
            btn.clicked.connect(
                lambda checked, k=key, b=btn: self._toggle_dropdown(k, None, b))
            tb.addWidget(btn)

        # ── Side dock panel ───────────────────────────────────────────────────
        side = self._build_side_panel()
        dock = QDockWidget("", self.main_window)
        dock.setWidget(side)
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        dock.rWidget(QWidget())   # remove title bar
        self.main_window.addDockWidget(Qt.LeftDockWidgetArea, dock)

# ── NBO / basis-file workflow ─────────────────────────────────────────────────

    def open_basis_file_dialog(self):
        """Entry point: pick a .47 or .31 basis file, then show matching key files."""
        fname, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Opes Set File",
            "",
            "NBO Basis Files (*.47 *.31);;All Files (*)"
        )
        if not fname:
            return
        dlg = _KeyFilePickerDialog(fname, self.main_window)
        dlg.cubes_ready.connect(self._load_computed_cubes)
        dlg.exec_()

    def _load_computed_cubes(self, cube_paths):
        """Receive a list of freshly-computed cube paths and load them."""
        first_new_idx = None
        for path in cube_paths:
            path = os.path.abspath(path)
            try:
                cube = self.read_cube(path)
                idx  = len(self.cube_files)
                self.cube_files.append(path)
                self.cubes.append(cube)
                if self.cube_list is not None:
                    self.cube_list.blockSignals(True)
                    self.cube_list.addItem(QListWidgetItem(os.path.basename(path)))
                    self.cube_list.blockSignals(False)
                if first_new_idx is None:
                    first_new_idx = idx
            except Exception as e:
                QMessageBox.warning(
                    self.main_window, "Load Error",
                    f"Could not load computed cube:\n{path}\n\n{e}")
        if first_new_idx is not None:
            if self.cube_list is not None:
                self.cube_list.setCurrentRow(first_new_idx)
            else:
                self.switch_cube(first_new_idx)


    def visualize(self):
        self.app = QApplication(sys.argv)
        self.main_window = QMainWindow()
        self.main_window.setWindowTitle("Cube File Visualizer")
        self.main_window.resize(1400, 900)
        self.main_window.setStyleSheet("QMainWindow { background:#1e1e2e; }")

        central = QWidget(); cl = QVBoxLayout(central); cl.setContentsMargins(0,0,0,0)
        self.plotter = QtInteractor(central)
        cl.addWidget(self.plotter.interactor)
        self.main_window.setCentralWidget(central)

        self.create_toolbar_and_side_panel()

        self.plotter.enable_point_picking(
            callback=self.atom_pick_callback,
            left_clicking=True,
            show_message="Left-click to select an atom.",
            show_point=False,
        )
        self.plotter.iren.interactor.AddObserver(
            "LeftButtonPressEvent", self._on_left_click
        )
        self.plotter.set_background(self.background_color)
        if self.cubes:
            self.update_visualization()
        self.plotter.show_axes()
        self.plotter.camera_position = "yz"
        self.plotter.camera.azimuth   = 30
        self.plotter.camera.elevation = 20
        self.main_window.show()
        sys.exit(self.app.exec_())


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    pv.global_theme.allow_empty_mesh = True
    parser = argparse.ArgumentParser(description="Visualize Gaussian cube files")
    parser.add_argument("cube_files", nargs='*',
                        help="Optional .cube files to load on startup. "
                             "More files can be dragged onto the window at any time.")
    args = parser.parse_args()
    valid = [f for f in args.cube_files if os.path.isfile(f)]
    if args.cube_files and not valid:
        print("No valid cube files found among the arguments provided.")
    MultiCubeVisualizer(valid).visualize()
