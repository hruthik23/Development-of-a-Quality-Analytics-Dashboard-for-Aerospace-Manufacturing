import sys
import os
import json
import logging
from logging.handlers import RotatingFileHandler
import traceback
import random
import sqlite3
import warnings
from collections import OrderedDict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta


import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from PyQt5.QtCore import (
    Qt, QDate, QThread, pyqtSignal, QStringListModel, QTimer, QSize,
    QModelIndex, QAbstractTableModel, QReadWriteLock, QCoreApplication,
    QPropertyAnimation, QEasingCurve
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFormLayout, QFileDialog, QPushButton, QTabWidget, QLabel, QComboBox,
    QMessageBox, QTextEdit, QGroupBox, QDateEdit, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QSpinBox, QCheckBox, QStatusBar, QTableView, QHeaderView,
    QSplitter, QAbstractItemView, QLineEdit, QListWidget, QListWidgetItem,
    QProgressBar, QFrame, QProgressDialog, QInputDialog, QStackedWidget,
    QToolButton, QToolBar, QSizePolicy, QScrollArea, QGraphicsDropShadowEffect,
    QSplashScreen, QDesktopWidget, QStyledItemDelegate, QStyle
)
from PyQt5.QtGui import (
    QStandardItemModel, QStandardItem, QColor, QBrush, QIcon, QPixmap,
    QFont, QPainter, QPalette, QLinearGradient, QBrush as QBrushG,
    QPen, QFontDatabase
)

# Optional dependencies
try:
    import mplcursors
    HAS_MPLCURSORS = True
except ImportError:
    HAS_MPLCURSORS = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Machine learning
try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# PDF export
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.lib.utils import ImageReader
    from io import BytesIO
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# Statsmodels
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# Enterprise SQL 
try:
    from sqlalchemy import create_engine
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

# Matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

# ----------------------------------------------------------------------
# Dynamic Chart Canvas
# ----------------------------------------------------------------------
class MplCanvas(FigureCanvas):
    """Forces Matplotlib canvases to expand entirely into the available layout space."""
    def __init__(self, figure):
        super().__init__(figure)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
APP_NAME = "Aeroanalytics"
APP_SUBTITLE = "Aerospace quality intelligence platform"
APP_VERSION = "8.2"
SETTINGS_FILE = "aeroanalytics_settings.json"
LOG_FILE = "aeroanalytics.log"
ICON_FILE = "aeroanalytics_icon.png"
FILTER_PRESETS_FILE = "filter_presets.json"
CACHE_MAX_SIZE = 32

# Column name constants
COL_DATE = "Date"
COL_SUPPLIER = "Supplier"
COL_PART_TYPE = "Part_Type"
COL_DEFECT_TYPE = "Defect_Type"
COL_DEFECT_SEVERITY = "Defect_Severity"
COL_MATERIAL = "MaterialComposition"
COL_DEFECT_COUNT = "DefectCount"
COL_UNITS_INSPECTED = "UnitsInspected"
COL_COST_IMPACT = "CostImpact"
COL_ON_TIME_FLAG = "OnTimeFlag"
COL_YIELD = "YieldPercentage"
COL_QUALITY_SCORE = "QualityScore"
COL_DEFECT_RATE = "DefectRate"

# Aliases
COLUMN_ALIASES = {
    "Cost_Impact_SGD": COL_COST_IMPACT,
    "SupplierID": COL_SUPPLIER,
    "SupplierName": COL_SUPPLIER,
    "PartType": COL_PART_TYPE,
    "DefectType": COL_DEFECT_TYPE,
    "Severity": COL_DEFECT_SEVERITY,
    "Material": COL_MATERIAL,
    "Yield": COL_YIELD,
    "QualityScore": COL_QUALITY_SCORE,
    "DefectRate": COL_DEFECT_RATE,
    "OnTimeFlag": COL_ON_TIME_FLAG,
    "DeliveryDelay": "DeliveryDelay",
    "InspectionDate": COL_DATE,
    "EventDate": COL_DATE,
    "CreatedDate": COL_DATE,
    "TxnDate": COL_DATE,
    "DefectDate": COL_DATE
}

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
handler = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
handler.setFormatter(formatter)

logger = logging.getLogger(APP_NAME)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')

def shorten_label(value: Any, max_len: int = 18) -> str:
    text = str(value)
    return text if len(text) <= max_len else text[:max_len-1] + '…'

def escape_html(text: str) -> str:
    import html
    return html.escape(text)

def sort_period_index(series: pd.Series) -> pd.Series:
    try:
        return series.sort_index()
    except Exception as exc:
        logger.exception("Sorting period index failed: %s", exc)
        return series

def downsample_labels(labels: List[str], max_labels: int) -> Tuple[List[int], List[str]]:
    if not labels:
        return [], []
    if len(labels) <= max_labels:
        return list(range(len(labels))), labels
    idx = np.linspace(0, len(labels)-1, max_labels, dtype=int)
    return idx.tolist(), [labels[i] for i in idx]

def quote_sqlite_identifier(identifier: str) -> str:
    if not identifier or not isinstance(identifier, str):
        raise ValueError("Invalid SQLite identifier.")
    return '"' + identifier.replace('"', '""') + '"'

def app_dir() -> Path:
    return Path(os.path.abspath(os.path.dirname(__file__)))

def locate_asset(filename: str) -> Optional[str]:
    candidates = [app_dir() / filename, Path.cwd() / filename]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
@dataclass
class AppSettings:
    supplier_weights: Dict[str, float] = field(
        default_factory=lambda: {"defect": 0.5, "delivery": 0.3, "cost": 0.2}
    )
    roi_defect_reduction_pct: float = 2.0
    monthly_resample_freq: str = "M"
    max_points_line_chart: int = 2500
    pca_sample_size: int = 1500
    enable_sampling: bool = True
    dark_mode: bool = True
    preferred_date_column: str = COL_DATE
    confidence_z_score: float = 1.96
    supplier_min_volume_for_ranking: int = 5
    rolling_window_points: int = 7
    max_category_labels: int = 10
    max_label_length: int = 16
    auto_refresh_ms: int = 350
    chart_padding: float = 2.0
    max_xtick_labels: int = 8

    alert_defect_rate_threshold: float = 5.0
    alert_copq_spike_threshold: float = 10000
    enable_realtime_monitoring: bool = False
    realtime_interval_ms: int = 5000

    anomaly_contamination: float = 0.1
    model_test_size: float = 0.3
    random_state: int = 42

    quality_index_winsorize: bool = True
    quality_index_winsor_limits: Tuple[float, float] = (0.05, 0.95)
    supplier_trend_window: int = 3
    forecast_method: str = "auto"
    anomaly_sample_size: int = 5000
    classifier_impute_strategy: str = "mean"
    classifier_max_categories: int = 50
    control_limits_ddof: int = 1
    enable_progress_dialogs: bool = True
    pdf_include_charts: bool = True

    # UI settings
    sidebar_collapsed: bool = False
    sidebar_width: int = 248
    collapsed_sidebar_width: int = 70
    animation_duration: int = 300

    def __post_init__(self):
        self.anomaly_contamination = min(max(self.anomaly_contamination, 0.01), 0.5)
        self.model_test_size = min(max(self.model_test_size, 0.1), 0.5)
        self.max_points_line_chart = max(100, self.max_points_line_chart)

class SettingsManager:
    def __init__(self, path: str = SETTINGS_FILE) -> None:
        self.path = Path(path)

    def load(self) -> AppSettings:
        if not self.path.exists():
            return AppSettings()
        try:
            payload = json.loads(self.path.read_text(encoding='utf-8'))
            default = asdict(AppSettings())
            default.update(payload)
            return AppSettings(**default)
        except Exception as exc:
            logger.exception("Failed to load settings: %s", exc)
            return AppSettings()

    def save(self, settings: AppSettings) -> None:
        try:
            self.path.write_text(json.dumps(asdict(settings), indent=2), encoding='utf-8')
        except Exception as exc:
            logger.exception("Failed to save settings: %s", exc)

# ----------------------------------------------------------------------
# Filter state and LRU cache
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class FilterState:
    part_type: str = "All Part Types"
    defect_type: str = "All Defect Types"
    severity: str = "All Severities"
    material: str = "All Materials"
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None

    def cache_key(self) -> Tuple[Any, ...]:
        return (
            self.part_type,
            self.defect_type,
            self.severity,
            self.material,
            self.start_date.isoformat() if self.start_date is not None else None,
            self.end_date.isoformat() if self.end_date is not None else None,
        )

class LRUCache:
    def __init__(self, maxsize: int = CACHE_MAX_SIZE):
        self.maxsize = maxsize
        self._cache = OrderedDict()

    def get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)

    def clear(self):
        self._cache.clear()

# ----------------------------------------------------------------------
# Data quality report
# ----------------------------------------------------------------------
@dataclass
class DataQualityReport:
    row_count: int = 0
    column_count: int = 0
    missing_by_column: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    detected_date_column: Optional[str] = None
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    duplicated_rows: int = 0
    negative_costs: int = 0
    zero_inspected: int = 0

    def to_text(self) -> str:
        lines = [
            f"Rows: {self.row_count:,}",
            f"Columns: {self.column_count}",
            f"Detected date column: {self.detected_date_column or 'None'}",
            f"Duplicate rows: {self.duplicated_rows:,}",
            "",
            "Key warnings:",
        ]
        if self.warnings:
            lines.extend([f"• {w}" for w in self.warnings[:12]])
        else:
            lines.append("• No major issues detected.")
        lines.append("")
        lines.append("Top missing-value columns:")
        top_missing = sorted(self.missing_by_column.items(), key=lambda x: x[1], reverse=True)[:10]
        if top_missing:
            lines.extend([f"• {col}: {count:,}" for col, count in top_missing])
        else:
            lines.append("• None")
        return "\n".join(lines)

# ----------------------------------------------------------------------
# Data validator
# ----------------------------------------------------------------------
class DataValidator:
    DATE_CANDIDATES = [COL_DATE, "InspectionDate", "EventDate", "CreatedDate", "TxnDate", "DefectDate"]

    @classmethod
    def detect_date_column(cls, df: pd.DataFrame) -> Optional[str]:
        for col in cls.DATE_CANDIDATES:
            if col in df.columns:
                return col
        for col in df.columns:
            if 'date' in col.lower():
                try:
                    pd.to_datetime(df[col], errors='raise')
                    return col
                except Exception:
                    continue
        return None

    @classmethod
    def normalize_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for src, dst in COLUMN_ALIASES.items():
            if dst not in df.columns and src in df.columns:
                df[dst] = df[src]

        date_col = cls.detect_date_column(df)
        if date_col and date_col != COL_DATE and COL_DATE not in df.columns:
            df[COL_DATE] = df[date_col]
        if COL_DATE in df.columns:
            df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors='coerce')
            if df[COL_DATE].dt.tz is not None:
                df[COL_DATE] = df[COL_DATE].dt.tz_convert(None)

        if COL_ON_TIME_FLAG not in df.columns and "DeliveryDelay" in df.columns:
            delay = safe_numeric(df["DeliveryDelay"])
            df[COL_ON_TIME_FLAG] = (delay.fillna(999999) <= 0).astype(float)

        if COL_DEFECT_COUNT not in df.columns and COL_DEFECT_TYPE in df.columns:
            df[COL_DEFECT_COUNT] = df[COL_DEFECT_TYPE].notna().astype(int)

        for col in [COL_UNITS_INSPECTED, COL_DEFECT_COUNT, COL_COST_IMPACT, COL_YIELD, COL_QUALITY_SCORE, COL_DEFECT_RATE]:
            if col in df.columns:
                df[col] = safe_numeric(df[col])

        if COL_UNITS_INSPECTED in df.columns and COL_DEFECT_COUNT in df.columns and COL_DEFECT_RATE not in df.columns:
            denom = df[COL_UNITS_INSPECTED].replace(0, np.nan)
            df[COL_DEFECT_RATE] = df[COL_DEFECT_COUNT] / denom

        if COL_COST_IMPACT in df.columns:
            neg = (df[COL_COST_IMPACT] < 0).sum()
            if neg > 0:
                warnings.warn(f"{neg} rows have negative {COL_COST_IMPACT}. They will be set to 0.")
                df.loc[df[COL_COST_IMPACT] < 0, COL_COST_IMPACT] = 0.0

        if COL_UNITS_INSPECTED in df.columns:
            zero = (df[COL_UNITS_INSPECTED] == 0).sum()
            if zero > 0:
                warnings.warn(f"{zero} rows have zero units inspected; defect rate will be missing.")

        return df

    @classmethod
    def build_report(cls, df: pd.DataFrame) -> DataQualityReport:
        report = DataQualityReport(
            row_count=len(df),
            column_count=len(df.columns),
            duplicated_rows=int(df.duplicated().sum()),
            detected_date_column=cls.detect_date_column(df),
        )
        report.missing_by_column = {col: int(df[col].isna().sum()) for col in df.columns}
        report.numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        report.categorical_columns = df.select_dtypes(exclude=[np.number, 'datetime64[ns]']).columns.tolist()

        if report.row_count == 0:
            report.warnings.append("Dataset is empty.")
        if report.detected_date_column is None:
            report.warnings.append("No date column detected. Trend analysis will be limited.")
        if report.duplicated_rows > 0:
            report.warnings.append(f"{report.duplicated_rows:,} duplicate rows detected.")

        essential_columns = [COL_DEFECT_TYPE, COL_SUPPLIER, COL_COST_IMPACT, COL_PART_TYPE]
        missing_essential = [col for col in essential_columns if col not in df.columns]
        if missing_essential:
            report.warnings.append(f"Missing common quality fields: {', '.join(missing_essential)}")

        high_missing = [
            col for col, count in report.missing_by_column.items()
            if report.row_count and count / report.row_count > 0.4
        ]
        if high_missing:
            report.warnings.append("High missingness (>40%) in: " + ", ".join(high_missing[:8]))

        if COL_COST_IMPACT in df.columns:
            report.negative_costs = int((df[COL_COST_IMPACT] < 0).sum())
            if report.negative_costs > 0:
                report.warnings.append(f"{report.negative_costs} rows have negative costs; set to 0.")

        if COL_UNITS_INSPECTED in df.columns:
            report.zero_inspected = int((df[COL_UNITS_INSPECTED] == 0).sum())
            if report.zero_inspected > 0:
                report.warnings.append(f"{report.zero_inspected} rows have zero units inspected; defect rate missing.")

        return report

# ----------------------------------------------------------------------
# Data store with thread-safe locking
# ----------------------------------------------------------------------
class QualityDataStore:
    def __init__(self) -> None:
        self._raw_df = pd.DataFrame()
        self._normalized_df = pd.DataFrame()
        self._quality_report = DataQualityReport()
        self._filter_cache = LRUCache(maxsize=CACHE_MAX_SIZE)
        self._lock = QReadWriteLock()
        self.db_connection = None
        self.db_table = None

    @property
    def raw_df(self) -> pd.DataFrame:
        self._lock.lockForRead()
        try:
            return self._raw_df
        finally:
            self._lock.unlock()

    @property
    def df(self) -> pd.DataFrame:
        self._lock.lockForRead()
        try:
            return self._normalized_df
        finally:
            self._lock.unlock()

    @property
    def quality_report(self) -> DataQualityReport:
        self._lock.lockForRead()
        try:
            return self._quality_report
        finally:
            self._lock.unlock()

    def load_file(self, file_path: str) -> pd.DataFrame:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(file_path)
        elif ext in (".csv", ".txt"):
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding="latin1")
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        self._lock.lockForWrite()
        try:
            self._raw_df = df.copy()
            self._normalized_df = DataValidator.normalize_columns(df)
            self._quality_report = DataValidator.build_report(self._normalized_df)
            self._filter_cache.clear()
            self.db_connection = None
            self.db_table = None
        finally:
            self._lock.unlock()
        return self._normalized_df

    def load_from_sqlite(self, db_path: str, table_name: str) -> pd.DataFrame:
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table_name,))
            if not cursor.fetchone():
                raise ValueError(f"Table '{table_name}' does not exist in database.")
            safe_table = quote_sqlite_identifier(table_name)
            df = pd.read_sql_query(f"SELECT * FROM {safe_table}", conn)
        finally:
            conn.close()

        self._lock.lockForWrite()
        try:
            self._raw_df = df.copy()
            self._normalized_df = DataValidator.normalize_columns(df)
            self._quality_report = DataValidator.build_report(self._normalized_df)
            self._filter_cache.clear()
            self.db_connection = db_path
            self.db_table = table_name
        finally:
            self._lock.unlock()
        return self._normalized_df

    def load_from_sql_uri(self, uri: str, query: str) -> pd.DataFrame:
        if not HAS_SQLALCHEMY:
            raise ImportError("SQLAlchemy is required for enterprise DB connections. Run: pip install sqlalchemy")
            
        self._lock.lockForWrite()
        try:
            engine = create_engine(uri)
            df = pd.read_sql_query(query, engine)
            
            self._raw_df = df.copy()
            self._normalized_df = DataValidator.normalize_columns(df)
            self._quality_report = DataValidator.build_report(self._normalized_df)
            self._filter_cache.clear()
            self.db_connection = uri
            self.db_table = "Enterprise_SQL_Query"
            return self._normalized_df
        finally:
            self._lock.unlock()

    def clear_cache(self) -> None:
        self._lock.lockForWrite()
        try:
            self._filter_cache.clear()
        finally:
            self._lock.unlock()

    def unique_values(self, column: str) -> List[str]:
        self._lock.lockForRead()
        try:
            if column not in self._normalized_df.columns:
                return []
            values = self._normalized_df[column].dropna().astype(str).str.strip()
            values = values[values != ""]
            return sorted(values.unique().tolist())
        finally:
            self._lock.unlock()

    def get_date_bounds(self, preferred: str = COL_DATE) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        self._lock.lockForRead()
        try:
            if self._normalized_df.empty:
                return None, None
            date_col = preferred if preferred in self._normalized_df.columns else DataValidator.detect_date_column(self._normalized_df)
            if not date_col:
                return None, None
            dates = pd.to_datetime(self._normalized_df[date_col], errors='coerce').dropna()
            if dates.empty:
                return None, None
            return dates.min(), dates.max()
        finally:
            self._lock.unlock()

    def apply_filters(self, state: FilterState, date_column: str = COL_DATE) -> pd.DataFrame:
        key = (state.cache_key(), date_column)
        self._lock.lockForRead()
        try:
            cached = self._filter_cache.get(key)
            if cached is not None:
                return cached.copy()
            df = self._normalized_df
        finally:
            self._lock.unlock()

        if df.empty:
            self._lock.lockForWrite()
            try:
                self._filter_cache.put(key, df)
            finally:
                self._lock.unlock()
            return df

        mask = pd.Series(True, index=df.index)

        def eq_filter(column: str, value: str, all_label: str) -> None:
            nonlocal mask
            if column in df.columns and value != all_label:
                mask &= df[column].astype(str) == value

        eq_filter(COL_PART_TYPE, state.part_type, "All Part Types")
        eq_filter(COL_DEFECT_TYPE, state.defect_type, "All Defect Types")
        eq_filter(COL_DEFECT_SEVERITY, state.severity, "All Severities")
        eq_filter(COL_MATERIAL, state.material, "All Materials")

        if date_column in df.columns and (state.start_date is not None or state.end_date is not None):
            dates = pd.to_datetime(df[date_column], errors='coerce')
            if state.start_date is not None:
                mask &= dates >= state.start_date
            if state.end_date is not None:
                mask &= dates <= state.end_date

        result = df.loc[mask].copy()
        self._lock.lockForWrite()
        try:
            self._filter_cache.put(key, result)
        finally:
            self._lock.unlock()
        return result

# ----------------------------------------------------------------------
# Async Workers
# ----------------------------------------------------------------------
class DataLoaderThread(QThread):
    data_loaded = pyqtSignal(pd.DataFrame)
    error_occurred = pyqtSignal(str)

    def __init__(self, store: QualityDataStore, file_path: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.store = store
        self.file_path = file_path

    def run(self) -> None:
        try:
            df = self.store.load_file(self.file_path)
            self.data_loaded.emit(df)
        except Exception as exc:
            logger.exception("Load failure:\n%s", traceback.format_exc())
            self.error_occurred.emit(str(exc))

class MLWorkerThread(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, task_func, *args, **kwargs):
        super().__init__()
        self.task_func = task_func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.task_func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            logger.exception("ML Task failure")
            self.error.emit(str(e))

# ----------------------------------------------------------------------
# Filter controller
# ----------------------------------------------------------------------
class FilterController:
    FILTER_DEFS = [
        (COL_PART_TYPE, "All Part Types"),
        (COL_DEFECT_TYPE, "All Defect Types"),
        (COL_DEFECT_SEVERITY, "All Severities"),
        (COL_MATERIAL, "All Materials"),
    ]

    def __init__(self, store: QualityDataStore, combos: Dict[str, QComboBox], date_from: QDateEdit, date_to: QDateEdit) -> None:
        self.store = store
        self.combos = combos
        self.date_from = date_from
        self.date_to = date_to
        self.models: Dict[str, QStringListModel] = {}
        for col, _ in self.FILTER_DEFS:
            model = QStringListModel()
            self.models[col] = model
            self.combos[col].setModel(model)

    def populate_from_store(self) -> None:
        for col, all_label in self.FILTER_DEFS:
            combo = self.combos[col]
            current = combo.currentText() or all_label
            values = [all_label] + self.store.unique_values(col)
            self.models[col].setStringList(values)
            idx = combo.findText(current)
            combo.setCurrentIndex(idx if idx >= 0 else 0)

        min_date, max_date = self.store.get_date_bounds()
        if min_date is not None and max_date is not None:
            self.date_from.setEnabled(True)
            self.date_to.setEnabled(True)
            self.date_from.setDate(QDate(min_date.year, min_date.month, min_date.day))
            self.date_to.setDate(QDate(max_date.year, max_date.month, max_date.day))

    def current_state(self) -> FilterState:
        start = pd.Timestamp(self.date_from.date().toPyDate()) if self.date_from.isEnabled() else None
        end = (
            pd.Timestamp(self.date_to.date().toPyDate()) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            if self.date_to.isEnabled() else None
        )
        return FilterState(
            part_type=self.combos[COL_PART_TYPE].currentText(),
            defect_type=self.combos[COL_DEFECT_TYPE].currentText(),
            severity=self.combos[COL_DEFECT_SEVERITY].currentText(),
            material=self.combos[COL_MATERIAL].currentText(),
            start_date=start,
            end_date=end,
        )

# ----------------------------------------------------------------------
# Efficient table model with sorting and conditional formatting
# ----------------------------------------------------------------------
class DataFrameModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._dataframe = pd.DataFrame()
        self._anomaly_flags = None
        self._color_columns = []
        self._col_ranges = {}

    def set_dataframe(self, df: pd.DataFrame, anomaly_flags: Optional[pd.Series] = None, color_columns: Optional[List[str]] = None):
        self.beginResetModel()
        self._dataframe = df.copy()
        self._anomaly_flags = anomaly_flags
        self._color_columns = color_columns or []
        self._col_ranges = {}
        
        for col in self._color_columns:
            if is_numeric_dtype(self._dataframe[col]):
                s = self._dataframe[col].dropna()
                if not s.empty:
                    self._col_ranges[col] = (s.min(), s.max())
                    
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self._dataframe)

    def columnCount(self, parent=QModelIndex()):
        return len(self._dataframe.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row = index.row()
        col = index.column()
        value = self._dataframe.iat[row, col]
        
        if role == Qt.DisplayRole:
            if pd.isna(value):
                return ""
            if isinstance(value, float):
                return f"{value:,.4f}"
            return str(value)
        elif role == Qt.BackgroundRole and self._anomaly_flags is not None:
            if row < len(self._anomaly_flags) and bool(self._anomaly_flags.iloc[row]):
                return QColor(73, 20, 35)
        elif role == Qt.BackgroundRole and self._color_columns:
            col_name = self._dataframe.columns[col]
            if col_name in self._col_ranges:
                c_min, c_max = self._col_ranges[col_name]
                if c_max > c_min and pd.notna(value):
                    norm = (value - c_min) / (c_max - c_min)
                    r = int(255 * (1 - norm))
                    g = int(255 * norm)
                    return QColor(r, g, 0, 100)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._dataframe.columns[section])
        else:
            return str(section + 1)

    def sort(self, column, order):
        if column >= 0 and not self._dataframe.empty and column < len(self._dataframe.columns):
            col_name = self._dataframe.columns[column]
            self.layoutAboutToBeChanged.emit()
            try:
                self._dataframe.sort_values(by=col_name, ascending=(order == Qt.AscendingOrder), inplace=True)
                if self._anomaly_flags is not None:
                    self._anomaly_flags = self._anomaly_flags.reindex(self._dataframe.index)
            except Exception as exc:
                logger.exception("Sorting failed: %s", exc)
            self.layoutChanged.emit()

class DataFrameTableWidget(QTableView):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._model = DataFrameModel(self)
        self.setModel(self._model)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().setVisible(False)
        self.setShowGrid(False)
        self.setFrameShape(QFrame.NoFrame)
        self.setSortingEnabled(True)
        self.horizontalHeader().setSortIndicatorShown(True)

    def set_dataframe(self, df: pd.DataFrame, highlight_anomalies: Optional[pd.Series] = None, color_columns: Optional[List[str]] = None) -> None:
        self._model.set_dataframe(df, highlight_anomalies, color_columns)

# ----------------------------------------------------------------------
# Plot manager with professional styling
# ----------------------------------------------------------------------
class PlotManager:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.palette = {}
        self.update_palette()

    def update_palette(self):
        if self.settings.dark_mode:
            self.palette.update({
                "bg": "#08111F", "panel": "#0F1A2E", "grid": "#2A4168", 
                "text": "#EAF2FF", "primary": "#2F6BFF", "secondary": "#1FD0B4",
                "warning": "#F5A524", "danger": "#FF6078", "muted": "#8EA3C7", "accent": "#28C7FA"
            })
        else:
            self.palette.update({
                "bg": "#F4F7FB", "panel": "#FFFFFF", "grid": "#E2E8F0", 
                "text": "#1E293B", "primary": "#2F6BFF", "secondary": "#1FD0B4",
                "warning": "#F5A524", "danger": "#FF6078", "muted": "#8EA3C7", "accent": "#28C7FA"
            })
        if HAS_SEABORN:
            sns.set_palette("husl")
            self.palette.update({f"color{i}": c for i, c in enumerate(sns.color_palette("husl", 10).as_hex())})

    def prepare(self, figure: Figure, nrows: int = 1, ncols: int = 1, clear: bool = True):
        if clear:
            figure.clear()
        figure.patch.set_facecolor(self.palette["bg"])
        axes = figure.subplots(nrows, ncols)
        if isinstance(axes, np.ndarray):
            for ax in axes.flat:
                self._style_ax(ax)
        else:
            self._style_ax(axes)
        return axes

    def _style_ax(self, ax) -> None:
        ax.set_facecolor(self.palette["panel"])
        ax.grid(True, alpha=0.24, color=self.palette["grid"])
        ax.tick_params(colors=self.palette["text"], labelsize=9)
        ax.xaxis.label.set_color(self.palette["text"])
        ax.yaxis.label.set_color(self.palette["text"])
        ax.title.set_color(self.palette["text"])
        if hasattr(ax, "xaxis") and not getattr(ax, "name", "") == "polar":
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
        if hasattr(ax, "yaxis") and not getattr(ax, "name", "") == "polar":
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        for spine in ax.spines.values():
            spine.set_color(self.palette["grid"])

    def draw_empty(self, figure: Figure, canvas: FigureCanvas, title: str, message: str) -> None:
        ax = self.prepare(figure)
        ax.text(0.5, 0.5, message, ha="center", va="center", color=self.palette["text"])
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        figure.tight_layout(pad=self.settings.chart_padding)
        canvas.draw_idle()

    def safe_barh(self, ax, labels: List[str], values: List[float], color: str, title: str, xlabel: str = "") -> None:
        if not labels:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", color=self.palette["text"])
            ax.set_title(title)
            return
        pairs = list(zip(labels, values))[:self.settings.max_category_labels]
        short_labels = [shorten_label(v, self.settings.max_label_length) for v, _ in pairs]
        short_values = [float(v) for _, v in pairs]
        y = np.arange(len(short_labels))
        ax.barh(y, short_values, color=color)
        ax.set_yticks(y)
        ax.set_yticklabels(short_labels)
        ax.invert_yaxis()
        ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        ax.margins(y=0.08)

    def safe_vertical_bars(self, ax, labels: List[str], values: List[float], color: str, title: str, ylabel: str = "") -> None:
        if not labels:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", color=self.palette["text"])
            ax.set_title(title)
            return
        pairs = list(zip(labels, values))[:self.settings.max_category_labels]
        plot_labels = [shorten_label(v, self.settings.max_label_length) for v, _ in pairs]
        plot_values = [float(v) for _, v in pairs]
        x = np.arange(len(plot_labels))
        ax.bar(x, plot_values, color=color)
        idx, tick_labels = downsample_labels(plot_labels, self.settings.max_xtick_labels)
        ax.set_xticks(idx)
        ax.set_xticklabels(tick_labels, rotation=25, ha="right")
        ax.set_title(title)
        if ylabel:
            ax.set_ylabel(ylabel)
        ax.margins(x=0.03)

    def heatmap(self, ax, matrix: pd.DataFrame, title: str, annotate: bool = False) -> None:
        if matrix.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", color=self.palette["text"])
            ax.set_title(title)
            return
        img = ax.imshow(matrix.values, aspect="auto", cmap="Blues")
        ax.figure.colorbar(img, ax=ax, shrink=0.8)
        ax.set_xticks(np.arange(len(matrix.columns)))
        ax.set_yticks(np.arange(len(matrix.index)))
        ax.set_xticklabels([shorten_label(c, self.settings.max_label_length) for c in matrix.columns], rotation=35, ha="right")
        ax.set_yticklabels([shorten_label(i, self.settings.max_label_length) for i in matrix.index])
        ax.set_title(title)
        if annotate and matrix.shape[0] <= 10 and matrix.shape[1] <= 10:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(j, i, str(matrix.iloc[i, j]), ha="center", va="center", color="black", fontsize=8)

    def radar_chart(self, ax, metrics: Dict[str, float], title: str) -> None:
        labels = list(metrics.keys())
        values = [float(v) for v in metrics.values()]
        if not labels:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", color=self.palette["text"])
            ax.set_title(title)
            return
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        ax.plot(angles, values, color=self.palette["accent"], linewidth=2)
        ax.fill(angles, values, color=self.palette["accent"], alpha=0.18)
        ax.set_thetagrids(np.degrees(angles[:-1]), [shorten_label(l, 12) for l in labels])
        ax.set_ylim(0, 1)
        ax.set_title(title, pad=18)

# ----------------------------------------------------------------------
# Analytics engine
# ----------------------------------------------------------------------
class AnalyticsEngine:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.anomaly_model = None
        self.classifier_model = None
        self.classifier_scaler = None
        self.classifier_label_encoder = None
        self.classifier_features: List[str] = []
        self.classifier_target = None
        self.classifier_accuracy = 0.0

    @staticmethod
    def count_defects(df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        if COL_DEFECT_COUNT in df.columns:
            return int(safe_numeric(df[COL_DEFECT_COUNT]).fillna(0).sum())
        if COL_DEFECT_TYPE in df.columns:
            return int(df[COL_DEFECT_TYPE].notna().sum())
        return len(df)

    @staticmethod
    def total_copq(df: pd.DataFrame) -> float:
        if COL_COST_IMPACT not in df.columns:
            return 0.0
        return float(safe_numeric(df[COL_COST_IMPACT]).fillna(0).sum())

    def compute_quality_index(self, df: pd.DataFrame) -> pd.Series:
        if df.empty:
            return pd.Series(dtype=float)

        metric_specs = {
            COL_DEFECT_RATE: "lower_better",
            COL_QUALITY_SCORE: "higher_better",
            COL_YIELD: "higher_better",
            COL_ON_TIME_FLAG: "higher_better",
        }
        available = [c for c in metric_specs if c in df.columns]
        if not available:
            return pd.Series(dtype=float)

        norm = pd.DataFrame(index=df.index)
        for col in available:
            s = safe_numeric(df[col])
            if self.settings.quality_index_winsorize:
                lower, upper = self.settings.quality_index_winsor_limits
                s = s.clip(lower=s.quantile(lower), upper=s.quantile(upper))
            s_min, s_max = s.min(), s.max()
            if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
                norm[col] = 0.5
                continue
            scaled = (s - s_min) / (s_max - s_min)
            if metric_specs[col] == "lower_better":
                scaled = 1 - scaled
            norm[col] = scaled.clip(0, 1)
        return norm.mean(axis=1) * 100

    def defect_pareto(self, df: pd.DataFrame) -> pd.DataFrame:
        if COL_DEFECT_TYPE not in df.columns:
            return pd.DataFrame(columns=["Count", "CumulativePct"])
        counts = df[COL_DEFECT_TYPE].fillna("Unknown").value_counts().head(self.settings.max_category_labels)
        pareto = pd.DataFrame({"Count": counts})
        pareto["CumulativePct"] = pareto["Count"].cumsum() / pareto["Count"].sum() * 100
        return pareto

    def trend_by_period(self, df: pd.DataFrame, freq: str = "D") -> pd.Series:
        if COL_DATE not in df.columns:
            return pd.Series(dtype=float)
        temp = df.copy()
        temp[COL_DATE] = pd.to_datetime(temp[COL_DATE], errors='coerce')
        temp = temp.dropna(subset=[COL_DATE])
        if temp.empty:
            return pd.Series(dtype=float)
        if COL_DEFECT_COUNT in temp.columns:
            s = temp.set_index(COL_DATE)[COL_DEFECT_COUNT].resample(freq).sum()
        else:
            s = temp.set_index(COL_DATE).resample(freq).size()
        return sort_period_index(s.fillna(0))

    def monthly_copq(self, df: pd.DataFrame) -> pd.Series:
        if COL_DATE not in df.columns or COL_COST_IMPACT not in df.columns:
            return pd.Series(dtype=float)
        temp = df[[COL_DATE, COL_COST_IMPACT]].copy()
        temp[COL_DATE] = pd.to_datetime(temp[COL_DATE], errors='coerce')
        temp[COL_COST_IMPACT] = safe_numeric(temp[COL_COST_IMPACT]).fillna(0)
        temp = temp.dropna(subset=[COL_DATE])
        if temp.empty:
            return pd.Series(dtype=float)
        return sort_period_index(temp.set_index(COL_DATE)[COL_COST_IMPACT].resample(self.settings.monthly_resample_freq).sum())

    def monthly_defect_heatmap(self, df: pd.DataFrame) -> pd.DataFrame:
        if COL_DATE not in df.columns or COL_DEFECT_SEVERITY not in df.columns:
            return pd.DataFrame()
        temp = df[[COL_DATE, COL_DEFECT_SEVERITY]].copy()
        temp[COL_DATE] = pd.to_datetime(temp[COL_DATE], errors='coerce')
        temp = temp.dropna(subset=[COL_DATE])
        if temp.empty:
            return pd.DataFrame()
        temp["Month"] = temp[COL_DATE].dt.to_period("M").astype(str)
        matrix = pd.crosstab(temp[COL_DEFECT_SEVERITY].fillna("Unknown"), temp["Month"])
        return matrix.iloc[:self.settings.max_category_labels, -12:]

    def defect_cost_by_part(self, df: pd.DataFrame) -> pd.DataFrame:
        if COL_PART_TYPE not in df.columns:
            return pd.DataFrame()
        defect_count = df.groupby(COL_PART_TYPE)[COL_DEFECT_COUNT].sum() if COL_DEFECT_COUNT in df.columns else df.groupby(COL_PART_TYPE).size()
        result = pd.DataFrame({"Defects": defect_count})
        if COL_COST_IMPACT in df.columns:
            result["COPQ"] = df.groupby(COL_PART_TYPE)[COL_COST_IMPACT].sum()
        return result.fillna(0).sort_values(by="Defects", ascending=False).head(self.settings.max_category_labels)

    def severity_part_heatmap(self, df: pd.DataFrame) -> pd.DataFrame:
        if COL_DEFECT_SEVERITY not in df.columns or COL_PART_TYPE not in df.columns:
            return pd.DataFrame()
        matrix = pd.crosstab(df[COL_DEFECT_SEVERITY].fillna("Unknown"), df[COL_PART_TYPE].fillna("Unknown"))
        return matrix.iloc[:self.settings.max_category_labels, :self.settings.max_category_labels]

    def root_cause_drilldown(self, df: pd.DataFrame) -> pd.DataFrame:
        group_cols = [c for c in [COL_DEFECT_TYPE, COL_PART_TYPE, COL_SUPPLIER, COL_DEFECT_SEVERITY] if c in df.columns]
        if len(group_cols) < 2:
            return pd.DataFrame()
        group = df.groupby(group_cols).size().reset_index(name="Count")
        return group.sort_values("Count", ascending=False).head(20)

    def root_cause_cooccurrence(self, df: pd.DataFrame) -> pd.DataFrame:
        columns = [c for c in [COL_DEFECT_TYPE, COL_PART_TYPE, COL_SUPPLIER, COL_DEFECT_SEVERITY] if c in df.columns]
        if len(columns) < 2:
            return pd.DataFrame()
        left_col, right_col = columns[0], columns[1]
        matrix = pd.crosstab(df[left_col].fillna("Unknown"), df[right_col].fillna("Unknown"))
        return matrix.iloc[:self.settings.max_category_labels, :self.settings.max_category_labels]

    def supplier_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        if COL_SUPPLIER not in df.columns:
            return pd.DataFrame()

        grouped = df.groupby(COL_SUPPLIER)
        size = grouped.size().rename("Volume")
        result = pd.DataFrame(size)

        if COL_DEFECT_COUNT in df.columns and COL_UNITS_INSPECTED in df.columns:
            sums = grouped[[COL_DEFECT_COUNT, COL_UNITS_INSPECTED]].sum(min_count=1)
            denom = sums[COL_UNITS_INSPECTED].replace(0, np.nan)
            defect_rate = (sums[COL_DEFECT_COUNT] / denom).fillna(0)
        else:
            counts = grouped[COL_DEFECT_TYPE].count() if COL_DEFECT_TYPE in df.columns else size
            defect_rate = (counts / counts.max()).fillna(0) if counts.max() else counts * 0

        result["DefectRate"] = defect_rate
        result["OnTimeRate"] = grouped[COL_ON_TIME_FLAG].mean().fillna(0) if COL_ON_TIME_FLAG in df.columns else np.nan
        result["AvgCostImpact"] = grouped[COL_COST_IMPACT].mean().fillna(0) if COL_COST_IMPACT in df.columns else np.nan

        def normalize(series: pd.Series, higher_better: bool) -> pd.Series:
            s = safe_numeric(series)
            s_min, s_max = s.min(), s.max()
            if pd.isna(s_min) or pd.isna(s_max) or s_min == s_max:
                return pd.Series(0.5, index=s.index)
            scaled = (s - s_min) / (s_max - s_min)
            return scaled if higher_better else 1 - scaled

        result["DefectScore"] = normalize(result["DefectRate"], higher_better=False)
        result["DeliveryScore"] = normalize(result["OnTimeRate"], higher_better=True)
        result["CostScore"] = normalize(result["AvgCostImpact"], higher_better=False)

        weights = self.settings.supplier_weights
        result["TotalScore"] = (
            result["DefectScore"].fillna(0.5) * weights.get("defect", 0.5)
            + result["DeliveryScore"].fillna(0.5) * weights.get("delivery", 0.3)
            + result["CostScore"].fillna(0.5) * weights.get("cost", 0.2)
        )

        n = result["Volume"].clip(lower=1)
        z = self.settings.confidence_z_score
        p = result["DefectRate"].clip(0, 1)
        margin = z * np.sqrt((p * (1 - p)) / n)
        result["DefectRateLower"] = (p - margin).clip(lower=0)
        result["DefectRateUpper"] = (p + margin).clip(upper=1)

        if COL_DATE in df.columns:
            cols = [COL_SUPPLIER, COL_DATE]
            if COL_DEFECT_COUNT in df.columns:
                cols.append(COL_DEFECT_COUNT)
            temp = df[cols].copy()
            temp[COL_DATE] = pd.to_datetime(temp[COL_DATE], errors='coerce')
            temp = temp.dropna(subset=[COL_DATE])
            temp["Period"] = temp[COL_DATE].dt.to_period("M")
            if COL_DEFECT_COUNT in temp.columns:
                monthly = temp.groupby([COL_SUPPLIER, "Period"])[COL_DEFECT_COUNT].sum().reset_index()
            else:
                monthly = temp.groupby([COL_SUPPLIER, "Period"]).size().reset_index(name=COL_DEFECT_COUNT)
            trend_map = {}
            for supplier, sdf in monthly.groupby(COL_SUPPLIER):
                sdf = sdf.sort_values("Period")
                if len(sdf) >= self.settings.supplier_trend_window:
                    recent = sdf.tail(self.settings.supplier_trend_window)
                    if recent[COL_DEFECT_COUNT].mean() > sdf.head(self.settings.supplier_trend_window)[COL_DEFECT_COUNT].mean():
                        trend_map[supplier] = "↑"
                    elif recent[COL_DEFECT_COUNT].mean() < sdf.head(self.settings.supplier_trend_window)[COL_DEFECT_COUNT].mean():
                        trend_map[supplier] = "↓"
                    else:
                        trend_map[supplier] = "→"
                else:
                    trend_map[supplier] = "→"
            result["TrendArrow"] = pd.Series(trend_map)
        else:
            result["TrendArrow"] = "→"

        result["RiskBand"] = pd.cut(
            result["TotalScore"],
            bins=[-0.01, 0.4, 0.7, 1.01],
            labels=["High Risk", "Moderate", "Preferred"],
        )
        result["RankEligible"] = result["Volume"] >= self.settings.supplier_min_volume_for_ranking
        return result.sort_values(["RankEligible", "TotalScore"], ascending=[False, False])

    def supplier_radar_metrics(self, supplier_row: pd.Series) -> Dict[str, float]:
        return {
            "Defect": float(supplier_row.get("DefectScore", 0)),
            "Delivery": float(supplier_row.get("DeliveryScore", 0)),
            "Cost": float(supplier_row.get("CostScore", 0)),
            "Total": float(supplier_row.get("TotalScore", 0)),
        }

    def copq_what_if(self, df: pd.DataFrame, reduction_pct: float) -> Dict[str, float]:
        baseline_copq = self.total_copq(df)
        defects = self.count_defects(df)
        if defects <= 0 or baseline_copq <= 0:
            return {
                "baseline_copq": baseline_copq,
                "reduction_pct": reduction_pct,
                "estimated_savings": 0.0,
                "projected_copq": baseline_copq,
            }
        savings = baseline_copq * (reduction_pct / 100.0)
        return {
            "baseline_copq": baseline_copq,
            "reduction_pct": reduction_pct,
            "estimated_savings": savings,
            "projected_copq": max(0.0, baseline_copq - savings),
        }

    def executive_kpis(self, df: pd.DataFrame) -> Dict[str, float]:
        kpis: Dict[str, float] = {}
        kpis["Rows"] = float(len(df))
        kpis["Defects"] = float(self.count_defects(df))
        kpis["COPQ"] = float(self.total_copq(df))
        if COL_ON_TIME_FLAG in df.columns:
            kpis["OnTimeRate"] = float(safe_numeric(df[COL_ON_TIME_FLAG]).mean() * 100)
        if COL_YIELD in df.columns:
            kpis["YieldPct"] = float(safe_numeric(df[COL_YIELD]).mean())
        qi = self.compute_quality_index(df)
        if not qi.empty:
            kpis["QualityIndex"] = float(qi.mean())
        return kpis

    def numeric_correlation(self, df: pd.DataFrame, max_rows: int) -> Tuple[pd.DataFrame, Optional[np.ndarray], float]:
        num = df.select_dtypes(include=[np.number]).dropna()
        if num.empty or num.shape[1] < 2:
            return pd.DataFrame(), None, 0.0
        if len(num) > max_rows:
            num = num.sample(n=max_rows, random_state=42)
        corr = num.corr().round(3)
        if not HAS_SKLEARN or num.shape[0] < 3:
            return corr, None, 0.0
        scaled = StandardScaler().fit_transform(num)
        pca = PCA(n_components=min(2, scaled.shape[1]))
        comp = pca.fit_transform(scaled)
        return corr, comp, float(pca.explained_variance_ratio_.sum() * 100)

    def control_limits(self, series: pd.Series) -> Dict[str, float]:
        if series.empty:
            return {"mean": 0.0, "ucl": 0.0, "lcl": 0.0}
        mean = float(series.mean())
        std = float(series.std(ddof=self.settings.control_limits_ddof))
        return {"mean": mean, "ucl": mean + 3 * std, "lcl": max(0.0, mean - 3 * std)}

    def forecast_copq(self, df: pd.DataFrame, months_ahead: int = 3) -> pd.Series:
        monthly = self.monthly_copq(df)
        if monthly.empty or len(monthly) < 3:
            return pd.Series(dtype=float)

        if self.settings.forecast_method == "linear" or not HAS_STATSMODELS:
            if not HAS_SKLEARN:
                return pd.Series(dtype=float)
            x = np.arange(len(monthly)).reshape(-1, 1)
            y = monthly.values
            model = LinearRegression()
            model.fit(x, y)
            future_x = np.arange(len(monthly), len(monthly) + months_ahead).reshape(-1, 1)
            forecast = model.predict(future_x)
        else:
            try:
                model = ARIMA(monthly, order=(1,1,0), trend='c')
                fitted = model.fit()
                forecast = fitted.forecast(steps=months_ahead)
            except Exception as exc:
                logger.exception("ARIMA model fit failed: %s", exc)
                return pd.Series(dtype=float)

        start = monthly.index[-1] + pd.offsets.MonthEnd(1)
        future_index = pd.date_range(start=start, periods=months_ahead, freq="ME")
        return pd.Series(forecast, index=future_index)

    def detect_anomalies(self, df: pd.DataFrame, progress_callback: Optional[Callable[[int, int], None]] = None) -> Optional[pd.Series]:
        if not HAS_SKLEARN:
            return None
        num_df = df.select_dtypes(include=[np.number]).dropna()
        if num_df.shape[0] < 10 or num_df.shape[1] < 1:
            return None

        sample_size = min(self.settings.anomaly_sample_size, len(num_df))
        if len(num_df) > sample_size:
            num_df = num_df.sample(n=sample_size, random_state=self.settings.random_state)

        model = IsolationForest(
            contamination=self.settings.anomaly_contamination,
            random_state=self.settings.random_state,
            n_jobs=-1
        )
        if progress_callback:
            progress_callback(0, 100)
        preds = model.fit_predict(num_df)
        if progress_callback:
            progress_callback(100, 100)
        self.anomaly_model = model
        flags = pd.Series(preds == -1, index=num_df.index)
        return flags.reindex(df.index, fill_value=False)

    def train_defect_classifier(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> Dict[str, Any]:
        if not HAS_SKLEARN:
            return {"success": False, "error": "scikit-learn not installed"}
        if not feature_cols:
            return {"success": False, "error": "No feature columns selected"}
        if target_col in feature_cols:
            return {"success": False, "error": "Target column cannot also be a feature"}

        missing = [c for c in feature_cols + [target_col] if c not in df.columns]
        if missing:
            return {"success": False, "error": f"Missing columns: {', '.join(missing)}"}

        model_df = df[feature_cols + [target_col]].copy()
        model_df = model_df.dropna(subset=[target_col])
        
        if model_df.empty:
            return {"success": False, "error": "Target column contains only missing values."}

        X = model_df[feature_cols]
        y = model_df[target_col].astype(str)

        cat_features = []
        num_features = []
        for col in feature_cols:
            if is_numeric_dtype(X[col]):
                num_features.append(col)
            else:
                cat_features.append(col)

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
            ]
        )

        imputer = SimpleImputer(strategy=self.settings.classifier_impute_strategy)
        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=self.settings.model_test_size,
            random_state=self.settings.random_state,
            stratify=y_enc if len(np.unique(y_enc)) > 1 else None,
        )

        X_train_trans = preprocessor.fit_transform(X_train)
        X_test_trans = preprocessor.transform(X_test)
        X_train_trans = imputer.fit_transform(X_train_trans)
        X_test_trans = imputer.transform(X_test_trans)

        clf = RandomForestClassifier(n_estimators=150, random_state=self.settings.random_state, n_jobs=-1)
        clf.fit(X_train_trans, y_train)
        y_pred = clf.predict(X_test_trans)
        accuracy = accuracy_score(y_test, y_pred)

        importances = clf.feature_importances_
        feature_names = preprocessor.get_feature_names_out()
        importance_dict = {}
        for name, imp in zip(feature_names, importances):
            if name.startswith('num__'):
                orig = name[5:]
                importance_dict[orig] = imp
            elif name.startswith('cat__'):
                orig = name.split('__')[1].split('_')[0] 
                importance_dict[orig] = importance_dict.get(orig, 0) + imp
        for orig in cat_features:
            if orig not in importance_dict:
                importance_dict[orig] = 0.0

        importance_series = pd.Series(importance_dict).sort_values(ascending=False)

        self.classifier_model = clf
        self.classifier_scaler = None
        self.classifier_label_encoder = le
        self.classifier_features = feature_cols
        self.classifier_target = target_col
        self.classifier_accuracy = accuracy

        return {
            "success": True,
            "accuracy": accuracy,
            "importance": importance_series.to_dict(),
            "classes": le.classes_.tolist(),
        }

# ----------------------------------------------------------------------
# UI Components (Space Optimized)
# ----------------------------------------------------------------------
class Surface(QFrame):
    def __init__(self, title: Optional[str] = None, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("surface")
        self.layout = QVBoxLayout(self)
        # Reduced Margins to give more space to the charts
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(4) 
        if title:
            label = QLabel(title)
            label.setObjectName("surfaceTitle")
            # Forces the label to "shrink wrap" and not push charts down
            label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum) 
            self.layout.addWidget(label)

class StatCard(QFrame):
    def __init__(self, title: str, value: str, accent: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("statCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(6)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("statCardTitle")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("statCardValue")
        accent_bar = QFrame()
        accent_bar.setFixedHeight(4)
        accent_bar.setStyleSheet(f"background:{accent}; border-radius:2px;")
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addStretch()
        layout.addWidget(accent_bar)

    def update_value(self, value: str) -> None:
        self.value_label.setText(value)

class PageFrame(QWidget):
    def __init__(self, plot_manager, page_title: str, page_subtitle: str = "", parent=None):
        super().__init__(parent)
        self.plot_manager = plot_manager
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(5) 
        
        self.title = QLabel(page_title)
        self.title.setObjectName("pageTitle")
        self.title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum) # Shrinkwrap
        
        self.subtitle = QLabel(page_subtitle)
        self.subtitle.setObjectName("pageSubtitle")
        self.subtitle.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum) # Shrinkwrap
        
        outer.addWidget(self.title)
        if page_subtitle:
            outer.addWidget(self.subtitle)
            
        self.body = QVBoxLayout()
        self.body.setSpacing(10)
        outer.addLayout(self.body)
        
    def render_empty_page(self, message: str) -> None:
        if hasattr(self, "info"):
            self.info.setPlainText(message)
        for fig_name, canvas_name in [("figure", "canvas"), ("pareto_figure", "pareto_canvas"), ("matrix_figure", "matrix_canvas"), ("rank_figure", "rank_canvas"), ("radar_figure", "radar_canvas")]:
            if hasattr(self, fig_name) and hasattr(self, canvas_name):
                self.plot_manager.draw_empty(getattr(self, fig_name), getattr(self, canvas_name), "No Data", message)

class InfoPanel(QTextEdit):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMinimumHeight(60)
        self.setMaximumHeight(120)  # Capped height so it doesn't squish charts
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.setFrameShape(QFrame.NoFrame)

class MessageCenter(QTextEdit):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setPlaceholderText("Operational messages, warnings, and data-quality notes appear here.")
        self.setMinimumHeight(80)
        self.setMaximumHeight(200) # Capped height so it acts as a true scrollable console
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.setFrameShape(QFrame.NoFrame)

    def post(self, level: str, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        color = {
            "info": "#9FD2FF",
            "warning": "#FFC16C",
            "error": "#FF7C8E",
            "alert": "#FF7C8E",
        }.get(level.lower(), "#DDE8FF")
        safe_msg = escape_html(message)
        self.append(f'<span style="color:{color};">[{timestamp}] [{level.upper()}] {safe_msg}</span>')

# ----------------------------------------------------------------------
# Page Implementations
# ----------------------------------------------------------------------
class ManagementPage(PageFrame):
    def __init__(self, plot_manager, parent=None):
        super().__init__(plot_manager, "Management Overview", "Executive quality, cost, and defect snapshot.", parent)
        surface = Surface()
        self.figure = Figure() 
        self.canvas = MplCanvas(self.figure)
        self.info = InfoPanel()
        surface.layout.addWidget(self.canvas)
        surface.layout.addWidget(self.info)
        self.body.addWidget(surface)

    def render(self, df: pd.DataFrame, analytics: AnalyticsEngine) -> None:
        if df.empty:
            self.render_empty_page("No data for current filter selection.")
            return
        kpis = analytics.executive_kpis(df)
        part = analytics.defect_cost_by_part(df)
        axes = self.plot_manager.prepare(self.figure, nrows=1, ncols=2)
        ax1, ax2 = axes[0], axes[1]
        self.plot_manager.safe_vertical_bars(ax1, list(kpis.keys()), list(kpis.values()), self.plot_manager.palette["primary"], "Executive KPI Snapshot", "Value")
        if not part.empty:
            self.plot_manager.safe_barh(ax2, part.index.astype(str).tolist(), part["Defects"].tolist(), self.plot_manager.palette["secondary"], "Defects by Part Type", "Defects")
        else:
            ax2.text(0.5, 0.5, "No part breakdown", ha="center", va="center", color=self.plot_manager.palette["text"])
            ax2.set_title("Defects by Part Type")
        self.figure.tight_layout(pad=self.plot_manager.settings.chart_padding)
        self.canvas.draw_idle()
        lines = ["Executive summary"] + [f"• {k}: {v:,.2f}" for k, v in kpis.items()]
        if not part.empty:
            lines.append(f"• Highest defect burden part type: {part.index[0]}")
        self.info.setPlainText("\n".join(lines))

class RootCausePage(PageFrame):
    def __init__(self, plot_manager, parent=None):
        super().__init__(plot_manager, "Root Cause Analysis", "Pareto, co-occurrence, and drilldown view.", parent)
        splitter = QSplitter(Qt.Vertical)
        top = QWidget()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        left_surface = Surface()
        right_surface = Surface()
        
        self.pareto_figure = Figure() 
        self.pareto_canvas = MplCanvas(self.pareto_figure)
        self.matrix_figure = Figure() 
        self.matrix_canvas = MplCanvas(self.matrix_figure)
        
        left_surface.layout.addWidget(self.pareto_canvas)
        right_surface.layout.addWidget(self.matrix_canvas)
        top_layout.addWidget(left_surface)
        top_layout.addWidget(right_surface)

        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        info_surface = Surface("Narrative")
        self.info = InfoPanel()
        info_surface.layout.addWidget(self.info)
        table_surface = Surface("Drilldown Table")
        self.table = DataFrameTableWidget()
        table_surface.layout.addWidget(self.table)
        bottom_layout.addWidget(info_surface, 2)
        bottom_layout.addWidget(table_surface, 3)

        splitter.addWidget(top)
        splitter.addWidget(bottom)
        
        # Priority to the charts!
        splitter.setStretchFactor(0, 4) 
        splitter.setStretchFactor(1, 1) 
        self.body.addWidget(splitter)

    def render(self, df: pd.DataFrame, analytics: AnalyticsEngine) -> None:
        pareto = analytics.defect_pareto(df)
        matrix = analytics.root_cause_cooccurrence(df)
        drilldown = analytics.root_cause_drilldown(df)

        ax1 = self.plot_manager.prepare(self.pareto_figure)
        if pareto.empty:
            ax1.text(0.5, 0.5, "No defect categories available", ha="center", va="center", color=self.plot_manager.palette["text"])
            ax1.set_title("Defect Pareto")
        else:
            labels = pareto.index.astype(str).tolist()
            values = pareto["Count"].tolist()
            x = np.arange(len(labels))
            ax1.bar(x, values, color=self.plot_manager.palette["primary"])
            idx, tick_labels = downsample_labels([shorten_label(x, self.plot_manager.settings.max_label_length) for x in labels], self.plot_manager.settings.max_xtick_labels)
            ax1.set_xticks(idx)
            ax1.set_xticklabels(tick_labels, rotation=22, ha="right")
            ax1.set_title("Defect Pareto")
            ax1.set_ylabel("Count")
            ax1_t = ax1.twinx()
            ax1_t.plot(x, pareto["CumulativePct"].tolist(), color=self.plot_manager.palette["danger"], marker="o")
            ax1_t.set_ylim(0, 110)
            ax1_t.set_ylabel("Cumulative %")
        self.pareto_figure.tight_layout(pad=self.plot_manager.settings.chart_padding)
        self.pareto_canvas.draw_idle()

        axm = self.plot_manager.prepare(self.matrix_figure)
        self.plot_manager.heatmap(axm, matrix, "Co-occurrence Matrix", annotate=True)
        self.matrix_figure.tight_layout(pad=self.plot_manager.settings.chart_padding)
        self.matrix_canvas.draw_idle()

        self.table.set_dataframe(drilldown)
        lines = ["Root cause explorer"]
        if not pareto.empty:
            for defect, row in pareto.head(5).iterrows():
                lines.append(f"• {defect}: {int(row['Count'])} events ({row['CumulativePct']:.1f}% cumulative)")
        if not drilldown.empty:
            lines.append("• Bottom table shows the strongest cross-dimension combinations.")
        self.info.setPlainText("\n".join(lines))

class TrendPage(PageFrame):
    def __init__(self, plot_manager, settings: AppSettings, parent=None):
        super().__init__(plot_manager, "Trend Analysis", "Control view of daily defect movement and time-based severity patterns.", parent)
        self.settings = settings
        surface = Surface()
        self.figure = Figure() 
        self.canvas = MplCanvas(self.figure)
        self.info = InfoPanel()
        surface.layout.addWidget(self.canvas)
        surface.layout.addWidget(self.info)
        self.body.addWidget(surface)

    def render(self, df: pd.DataFrame, analytics: AnalyticsEngine) -> None:
        series = analytics.trend_by_period(df, freq="D")
        heatmap = analytics.monthly_defect_heatmap(df)
        if series.empty:
            self.render_empty_page("No usable date series detected.")
            return

        axes = self.plot_manager.prepare(self.figure, nrows=2, ncols=1)
        ax1, ax2 = axes[0], axes[1]
        x = pd.Series(series.index)
        y = pd.Series(series.values)
        if len(y) > self.settings.max_points_line_chart:
            idx = np.linspace(0, len(y) - 1, self.settings.max_points_line_chart, dtype=int)
            x = x.iloc[idx]
            y = y.iloc[idx]
        ax1.plot(x, y, marker="o", markersize=2.8, linewidth=1.4, color=self.plot_manager.palette["primary"], label="Daily defects")
        rolling = series.rolling(window=max(2, self.settings.rolling_window_points), min_periods=1).mean()
        ax1.plot(rolling.index, rolling.values, linewidth=2.0, color=self.plot_manager.palette["secondary"], label="Rolling avg")
        limits = analytics.control_limits(series)
        ax1.axhline(limits["mean"], linestyle="--", color=self.plot_manager.palette["warning"], label="Mean")
        ax1.axhline(limits["ucl"], linestyle=":", color=self.plot_manager.palette["danger"], label="UCL")
        ax1.axhline(limits["lcl"], linestyle=":", color=self.plot_manager.palette["muted"], label="LCL")
        ax1.set_title("Defect Trend and Control Limits")
        ax1.set_ylabel("Defects")
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax1.legend(loc="upper right", fontsize=8)
        self.plot_manager.heatmap(ax2, heatmap, "Severity Heatmap Over Time", annotate=False)

        self.figure.autofmt_xdate(rotation=20)
        self.figure.tight_layout(pad=self.plot_manager.settings.chart_padding)
        self.canvas.draw_idle()

        out_of_control = int(((series > limits["ucl"]) | (series < limits["lcl"])).sum())
        lines = [
            f"Observations: {len(series)}",
            f"Mean daily defects: {limits['mean']:.2f}",
            f"Out-of-control points: {out_of_control}",
        ]
        if HAS_STATSMODELS and len(series) >= 24:
            try:
                decomp = seasonal_decompose(series, period=min(30, max(2, len(series) // 3)), model="additive")
                lines.append(f"Seasonal decomposition available. Median trend component: {np.nanmedian(decomp.trend):.2f}")
            except Exception:
                lines.append("Seasonal decomposition skipped due to decomposition error.")
        self.info.setPlainText("\n".join(lines))

class SupplierPage(PageFrame):
    def __init__(self, plot_manager, parent=None):
        super().__init__(plot_manager, "Supplier Intelligence", "Benchmark supplier risk, delivery, and cost impact.", parent)
        splitter = QSplitter(Qt.Vertical)

        top = QWidget()
        top_layout = QHBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        rank_surface = Surface("Supplier Ranking")
        radar_surface = Surface("Top Supplier Radar")
        
        self.rank_figure = Figure() 
        self.rank_canvas = MplCanvas(self.rank_figure)
        self.radar_figure = Figure() 
        self.radar_canvas = MplCanvas(self.radar_figure)
        
        rank_surface.layout.addWidget(self.rank_canvas)
        radar_surface.layout.addWidget(self.radar_canvas)
        top_layout.addWidget(rank_surface, 3)
        top_layout.addWidget(radar_surface, 2)

        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        info_surface = Surface("Narrative")
        self.info = InfoPanel()
        info_surface.layout.addWidget(self.info)
        table_surface = Surface("Supplier Detail")
        self.table = DataFrameTableWidget()
        table_surface.layout.addWidget(self.table)
        bottom_layout.addWidget(info_surface, 2)
        bottom_layout.addWidget(table_surface, 3)

        splitter.addWidget(top)
        splitter.addWidget(bottom)
        
        # Priority to the charts!
        splitter.setStretchFactor(0, 4) 
        splitter.setStretchFactor(1, 1) 
        self.body.addWidget(splitter)

    def render(self, df: pd.DataFrame, analytics: AnalyticsEngine) -> None:
        scores = analytics.supplier_scores(df)
        if scores.empty:
            self.render_empty_page("Supplier analysis unavailable because the Supplier column is missing.")
            self.table.set_dataframe(pd.DataFrame())
            return

        top_scores = scores.head(self.plot_manager.settings.max_category_labels)
        ax = self.plot_manager.prepare(self.rank_figure)
        labels = top_scores.index.astype(str).tolist()
        values = top_scores["TotalScore"].tolist()
        colors = [
            self.plot_manager.palette["secondary"] if str(b) == "Preferred"
            else self.plot_manager.palette["warning"] if str(b) == "Moderate"
            else self.plot_manager.palette["danger"]
            for b in top_scores["RiskBand"].tolist()
        ]
        self.plot_manager.safe_barh(ax, labels, values, self.plot_manager.palette["primary"], "Supplier Ranking", "Score")
        for patch, color in zip(ax.patches, colors):
            patch.set_color(color)
        self.rank_figure.tight_layout(pad=self.plot_manager.settings.chart_padding)
        self.rank_canvas.draw_idle()

        self.radar_figure.clear()
        
        # Radar Background Theme fix applied here!
        self.radar_figure.patch.set_facecolor(self.plot_manager.palette["bg"]) 
        
        radar_ax = self.radar_figure.add_subplot(111, polar=True)
        self.plot_manager._style_ax(radar_ax)
        self.plot_manager.radar_chart(radar_ax, analytics.supplier_radar_metrics(top_scores.iloc[0]), f"Top Supplier: {shorten_label(top_scores.index[0], 14)}")
        self.radar_figure.tight_layout(pad=self.plot_manager.settings.chart_padding)
        self.radar_canvas.draw_idle()

        lines = ["Supplier scoring summary"]
        for supplier, row in top_scores.head(5).iterrows():
            lines.append(
                f"• {supplier}: score={row['TotalScore']:.3f}, volume={int(row['Volume'])}, "
                f"trend={row['TrendArrow']}, CI=({row['DefectRateLower']:.3f}, {row['DefectRateUpper']:.3f}), "
                f"band={row['RiskBand']}"
            )
        self.info.setPlainText("\n".join(lines))

        table_df = top_scores[["Volume", "DefectRate", "OnTimeRate", "AvgCostImpact", "TotalScore", "TrendArrow", "RiskBand", "RankEligible"]].reset_index().rename(columns={"index": "Supplier"})
        self.table.set_dataframe(table_df)

class CopqPage(PageFrame):
    def __init__(self, plot_manager, parent=None):
        super().__init__(plot_manager, "COPQ and Forecasting", "Cost of poor quality with scenario modelling and forward view.", parent)
        surface = Surface()
        self.figure = Figure() 
        self.canvas = MplCanvas(self.figure)
        self.info = InfoPanel()
        surface.layout.addWidget(self.canvas)
        surface.layout.addWidget(self.info)
        self.body.addWidget(surface)

    def render(self, df: pd.DataFrame, analytics: AnalyticsEngine) -> None:
        monthly = analytics.monthly_copq(df)
        forecast = analytics.forecast_copq(df, months_ahead=3)
        what_if = analytics.copq_what_if(df, analytics.settings.roi_defect_reduction_pct)

        if monthly.empty and what_if["baseline_copq"] == 0:
            self.render_empty_page("No CostImpact data available for COPQ analysis.")
            return

        axes = self.plot_manager.prepare(self.figure, nrows=2, ncols=1)
        ax1, ax2 = axes[0], axes[1]

        if not monthly.empty:
            ax1.bar(monthly.index, monthly.values, color=self.plot_manager.palette["danger"], alpha=0.85, width=20)
            ax1.set_title("Monthly COPQ")
            ax1.set_ylabel("Cost")
            ax1.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            if not forecast.empty:
                ax1.plot(forecast.index, forecast.values, marker="o", linestyle="--", color=self.plot_manager.palette["secondary"], label="3-month forecast")
                ax1.legend(loc="upper left", fontsize=8)
        else:
            ax1.text(0.5, 0.5, "No monthly COPQ series available", ha="center", va="center", color=self.plot_manager.palette["text"])
            ax1.set_title("Monthly COPQ")

        scenario_labels = ["Baseline COPQ", "Projected COPQ", "Savings"]
        scenario_values = [what_if["baseline_copq"], what_if["projected_copq"], what_if["estimated_savings"]]
        colors = [self.plot_manager.palette["danger"], self.plot_manager.palette["secondary"], self.plot_manager.palette["accent"]]
        self.plot_manager.safe_vertical_bars(ax2, scenario_labels, scenario_values, colors[0], "What-if Improvement Scenario", "Cost")
        for patch, color in zip(ax2.patches, colors):
            patch.set_color(color)

        self.figure.autofmt_xdate(rotation=20)
        self.figure.tight_layout(pad=self.plot_manager.settings.chart_padding)
        self.canvas.draw_idle()

        lines = [
            f"Baseline COPQ: {what_if['baseline_copq']:,.0f}",
            f"What-if defect reduction: {what_if['reduction_pct']:.1f}%",
            f"Estimated savings: {what_if['estimated_savings']:,.0f}",
            f"Projected COPQ after improvement: {what_if['projected_copq']:,.0f}",
        ]
        if not forecast.empty:
            lines.append(f"• Forecast next period average COPQ: {forecast.mean():,.0f}")
        self.info.setPlainText("\n".join(lines))

class QualityIndexPage(PageFrame):
    def __init__(self, plot_manager, settings: AppSettings, parent=None):
        super().__init__(plot_manager, "Quality Index", "Composite quality health built from standardized operational metrics.", parent)
        self.settings = settings
        surface = Surface()
        self.figure = Figure() 
        self.canvas = MplCanvas(self.figure)
        self.info = InfoPanel()
        surface.layout.addWidget(self.canvas)
        surface.layout.addWidget(self.info)
        self.body.addWidget(surface)

    def render(self, df: pd.DataFrame, analytics: AnalyticsEngine) -> None:
        qi = analytics.compute_quality_index(df)
        if qi.empty:
            self.render_empty_page("Not enough standardized quality metrics to compute an index.")
            return

        axes = self.plot_manager.prepare(self.figure, nrows=2, ncols=1)
        ax1, ax2 = axes[0], axes[1]

        if COL_DATE in df.columns:
            plot_df = pd.DataFrame({COL_DATE: pd.to_datetime(df[COL_DATE], errors='coerce'), "QI": qi}).dropna().sort_values(COL_DATE)
            if len(plot_df) > self.settings.max_points_line_chart:
                plot_df = plot_df.iloc[np.linspace(0, len(plot_df) - 1, self.settings.max_points_line_chart, dtype=int)]
            ax1.plot(plot_df[COL_DATE], plot_df["QI"], color=self.plot_manager.palette["secondary"], linewidth=1.8)
            ax1.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        else:
            ax1.plot(qi.iloc[:self.settings.max_points_line_chart].index, qi.iloc[:self.settings.max_points_line_chart].values, color=self.plot_manager.palette["secondary"], linewidth=1.8)

        ax1.set_title("Composite Quality Index")
        ax1.set_ylabel("Index (0-100)")
        hist = qi.dropna()
        ax2.hist(hist.values, bins=20, color=self.plot_manager.palette["primary"], alpha=0.85)
        ax2.set_title("Quality Index Distribution")
        ax2.set_xlabel("Index")
        ax2.set_ylabel("Frequency")

        self.figure.autofmt_xdate(rotation=20)
        self.figure.tight_layout(pad=self.plot_manager.settings.chart_padding)
        self.canvas.draw_idle()

        self.info.setPlainText("\n".join([
            f"Mean QI: {qi.mean():.2f}",
            f"Median QI: {qi.median():.2f}",
            f"Min QI: {qi.min():.2f}",
            f"Max QI: {qi.max():.2f}",
        ]))

class AuditPage(PageFrame):
    def __init__(self, plot_manager, parent=None):
        super().__init__(plot_manager, "Audit Readiness", "Focus audit preparation on the highest concentration risk areas.", parent)
        surface = Surface()
        self.figure = Figure() 
        self.canvas = MplCanvas(self.figure)
        self.info = InfoPanel()
        surface.layout.addWidget(self.canvas)
        surface.layout.addWidget(self.info)
        self.body.addWidget(surface)

    def render(self, df: pd.DataFrame, analytics: AnalyticsEngine, report: DataQualityReport) -> None:
        matrix = analytics.severity_part_heatmap(df)
        if matrix.empty:
            self.render_empty_page("Heatmap unavailable. Need Defect_Severity and Part_Type.")
            return

        ax = self.plot_manager.prepare(self.figure)
        self.plot_manager.heatmap(ax, matrix, "Audit Focus Areas: Severity vs Part Type", annotate=True)
        self.figure.tight_layout(pad=self.plot_manager.settings.chart_padding)
        self.canvas.draw_idle()

        max_idx = np.unravel_index(np.argmax(matrix.values), matrix.shape)
        highest = (matrix.index[max_idx[0]], matrix.columns[max_idx[1]], int(matrix.values[max_idx]))
        lines = [
            "Audit readiness summary",
            f"• Highest concentration area: severity '{highest[0]}' / part '{highest[1]}' = {highest[2]} issues",
        ]
        for warning in report.warnings[:5]:
            lines.append(f"• Data quality note: {warning}")
        self.info.setPlainText("\n".join(lines))

class AdvancedPage(PageFrame):
    def __init__(self, plot_manager, settings: AppSettings, parent=None):
        super().__init__(plot_manager, "Advanced Analytics", "Correlation structure and PCA view of numeric fields.", parent)
        self.settings = settings
        surface = Surface()
        self.figure = Figure() 
        self.canvas = MplCanvas(self.figure)
        self.refresh_button = QPushButton("Refresh Correlation / PCA")
        self.info = InfoPanel()
        surface.layout.addWidget(self.canvas)
        surface.layout.addWidget(self.refresh_button)
        surface.layout.addWidget(self.info)
        self.body.addWidget(surface)

    def render(self, df: pd.DataFrame, analytics: AnalyticsEngine) -> None:
        corr, pca_data, explained = analytics.numeric_correlation(df, max_rows=self.settings.pca_sample_size)
        if corr.empty:
            self.render_empty_page("At least two sufficiently populated numeric columns are required.")
            return

        axes = self.plot_manager.prepare(self.figure, nrows=1, ncols=2)
        ax1, ax2 = axes[0], axes[1]
        self.plot_manager.heatmap(ax1, corr.iloc[:self.settings.max_category_labels, :self.settings.max_category_labels], "Numeric Correlation", annotate=False)

        if pca_data is not None and len(pca_data) > 0:
            ax2.scatter(pca_data[:, 0], pca_data[:, 1], alpha=0.65, color=self.plot_manager.palette["primary"])
            ax2.set_title(f"PCA Projection ({explained:.1f}% variance)")
            ax2.set_xlabel("PC1")
            ax2.set_ylabel("PC2")
        else:
            ax2.text(0.5, 0.5, "PCA unavailable", ha="center", va="center", color=self.plot_manager.palette["text"])
            ax2.set_title("PCA Projection")

        self.figure.tight_layout(pad=self.plot_manager.settings.chart_padding)
        self.canvas.draw_idle()

        corr_pairs = corr.where(~np.eye(corr.shape[0], dtype=bool)).stack().sort_values(key=lambda s: s.abs(), ascending=False)
        seen = set()
        lines = ["Correlation insights"]
        for (a, b), value in corr_pairs.items():
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"• {a} vs {b}: r={value:.3f}")
            if len(lines) >= 6:
                break
        self.info.setPlainText("\n".join(lines))

class AnomalyPage(PageFrame):
    def __init__(self, plot_manager, analytics: AnalyticsEngine, parent=None):
        super().__init__(plot_manager, "Anomaly Detection", "Surface numeric outliers and review flagged rows quickly.", parent)
        self.analytics = analytics
        controls = Surface()
        row = QHBoxLayout()
        self.detect_btn = QPushButton("Run Anomaly Detection")
        row.addWidget(self.detect_btn)
        row.addStretch()
        controls.layout.addLayout(row)

        table_surface = Surface("Anomaly Preview")
        self.table = DataFrameTableWidget()
        table_surface.layout.addWidget(self.table)

        narrative_surface = Surface("Narrative")
        self.info = InfoPanel()
        narrative_surface.layout.addWidget(self.info)

        self.body.addWidget(controls)
        self.body.addWidget(table_surface)
        self.body.addWidget(narrative_surface)

        self.current_df = pd.DataFrame()
        self.anomaly_flags = None
        self.detect_btn.clicked.connect(self.run_anomaly)

    def set_data(self, df: pd.DataFrame):
        self.current_df = df
        self.table.set_dataframe(df.head(100))

    def run_anomaly(self):
        if self.current_df.empty:
            self.info.setPlainText("No data to analyze.")
            return

        if self.plot_manager.settings.enable_progress_dialogs:
            progress = QProgressDialog("Detecting anomalies...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(500)

            def update_progress(current, total):
                if progress.wasCanceled():
                    raise Exception("Cancelled by user")
                progress.setValue(int(current / total * 100))
                QCoreApplication.processEvents()
        else:
            progress = None
            update_progress = None

        try:
            flags = self.analytics.detect_anomalies(self.current_df, progress_callback=update_progress if progress else None)
            if flags is None:
                self.info.setPlainText("Anomaly detection failed. Need at least 10 complete numeric rows and scikit-learn installed.")
                return
            self.anomaly_flags = flags
            preview_df = self.current_df.head(1000).copy()
            preview_flags = flags.loc[preview_df.index]
            self.table.set_dataframe(preview_df, highlight_anomalies=preview_flags)
            n_anomalies = int(flags.sum())
            self.info.setPlainText(
                f"Detected {n_anomalies} anomalies ({n_anomalies / len(flags) * 100:.1f}%). "
                f"Table preview limited to first {len(preview_df)} rows."
            )
        except Exception as e:
            logger.exception("Error during anomaly detection: %s", e)
            self.info.setPlainText(f"Error during anomaly detection: {e}")
        finally:
            if progress:
                progress.close()

class PredictionPage(PageFrame):
    def __init__(self, plot_manager, analytics: AnalyticsEngine, parent=None):
        super().__init__(plot_manager, "Predictive Model", "Train a defect classifier from selected numeric features.", parent)
        self.analytics = analytics
        top = Surface("Model Configuration")
        # Ensure configuration doesn't eat the screen space
        top.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        
        form = QFormLayout()
        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.feature_list.setMaximumHeight(120)
        form.addRow("Features (numeric/categorical)", self.feature_list)
        self.target_combo = QComboBox()
        form.addRow("Target", self.target_combo)
        self.train_btn = QPushButton("Train Model")
        form.addRow(self.train_btn)
        top.layout.addLayout(form)

        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        result_surface = Surface("Model Output")
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFrameShape(QFrame.NoFrame)
        result_surface.layout.addWidget(self.results_text)
        
        chart_surface = Surface("Feature Importance")
        self.figure = Figure() 
        self.canvas = MplCanvas(self.figure)
        chart_surface.layout.addWidget(self.canvas)
        
        bottom_layout.addWidget(result_surface, 2)
        bottom_layout.addWidget(chart_surface, 3)

        self.body.addWidget(top)
        self.body.addWidget(bottom)

        self.current_df = pd.DataFrame()
        self.train_btn.clicked.connect(self.train_model)

    def set_data(self, df: pd.DataFrame):
        self.current_df = df
        self.feature_list.clear()
        for col in df.columns:
            if col == COL_DATE:
                continue
            item = QListWidgetItem(col)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.feature_list.addItem(item)

        candidate_targets = []
        for c in df.columns:
            if c == COL_DATE:
                continue
            nunique = df[c].nunique(dropna=True)
            if 2 <= nunique <= self.plot_manager.settings.classifier_max_categories:
                candidate_targets.append(c)

        self.target_combo.clear()
        self.target_combo.addItems(candidate_targets)
        if not candidate_targets:
            self.results_text.setPlainText("No suitable categorical target columns found for classification.")
        else:
            self.results_text.clear()

    def train_model(self):
        if self.current_df.empty:
            self.results_text.setPlainText("No data.")
            return

        features = []
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            if item.checkState() == Qt.Checked:
                features.append(item.text())

        if len(features) < 1:
            self.results_text.setPlainText("Select at least one feature.")
            return

        target = self.target_combo.currentText()
        if not target:
            self.results_text.setPlainText("Select a valid target column.")
            return

        self.train_btn.setEnabled(False)
        self.results_text.setPlainText("Training model in background... Please wait.")

        self.worker = MLWorkerThread(
            self.analytics.train_defect_classifier, 
            self.current_df, features, target
        )
        self.worker.finished.connect(self._on_training_finished)
        self.worker.error.connect(self._on_training_error)
        self.worker.start()

    def _on_training_finished(self, result):
        self.train_btn.setEnabled(True)
        if not result["success"]:
            self.results_text.setPlainText(f"Training failed: {result.get('error', 'unknown error')}")
            return

        text = f"Model trained successfully!\nAccuracy: {result['accuracy']:.3f}\n\nFeature Importance:\n"
        for feat, imp in result["importance"].items():
            text += f"{feat}: {imp:.4f}\n"
        self.results_text.setPlainText(text)

        ax = self.plot_manager.prepare(self.figure)
        imp_series = pd.Series(result["importance"]).sort_values()
        ax.barh(imp_series.index, imp_series.values, color=self.plot_manager.palette["primary"])
        ax.set_title("Feature Importance")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _on_training_error(self, err_msg):
        self.train_btn.setEnabled(True)
        logger.error("Predictive Model Error: %s", err_msg)
        self.results_text.setPlainText(f"Fatal error during training: {err_msg}")

# ----------------------------------------------------------------------
# Helper dialogs
# ----------------------------------------------------------------------
class SQLConnectDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Connect to Enterprise Database")
        self.resize(500, 250)
        layout = QFormLayout(self)

        self.uri_edit = QLineEdit()
        self.uri_edit.setPlaceholderText("postgresql://user:pass@localhost:5432/dbname")
        layout.addRow("Connection URI:", self.uri_edit)

        self.query_edit = QTextEdit()
        self.query_edit.setPlaceholderText("SELECT * FROM quality_data")
        layout.addRow("SQL Query:", self.query_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_selection(self):
        return self.uri_edit.text(), self.query_edit.toPlainText()

class DatabaseDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Connect to SQLite Database")
        self.resize(420, 220)
        layout = QFormLayout(self)

        self.db_path_edit = QLineEdit()
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self.browse_db)
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.db_path_edit)
        path_layout.addWidget(browse_btn)
        layout.addRow("Database file:", path_layout)

        self.table_combo = QComboBox()
        layout.addRow("Table:", self.table_combo)

        self.refresh_btn = QPushButton("Refresh Tables")
        self.refresh_btn.clicked.connect(self.load_tables)
        layout.addRow("", self.refresh_btn)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def browse_db(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select SQLite Database", "", "SQLite DB (*.db *.sqlite *.sqlite3)")
        if path:
            self.db_path_edit.setText(path)
            self.load_tables()

    def load_tables(self):
        path = self.db_path_edit.text()
        if not os.path.exists(path):
            QMessageBox.warning(self, "Error", "Database file does not exist.")
            return
        try:
            conn = sqlite3.connect(path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            self.table_combo.clear()
            self.table_combo.addItems(tables)
        except Exception as e:
            logger.exception("Failed reading SQL tables: %s", e)
            QMessageBox.critical(self, "Error", f"Could not read tables: {e}")

    def get_selection(self):
        return self.db_path_edit.text(), self.table_combo.currentText()

class SettingsDialog(QDialog):
    def __init__(self, parent: Optional[QWidget], settings: AppSettings) -> None:
        super().__init__(parent)
        self.setWindowTitle("Advanced Settings")
        self.resize(600, 700)
        self._settings = settings
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)

        general_tab = QWidget()
        general_layout = QFormLayout(general_tab)
        self.defect_weight = QDoubleSpinBox(); self.defect_weight.setRange(0, 100); self.defect_weight.setValue(settings.supplier_weights.get("defect", 0.5) * 100)
        self.delivery_weight = QDoubleSpinBox(); self.delivery_weight.setRange(0, 100); self.delivery_weight.setValue(settings.supplier_weights.get("delivery", 0.3) * 100)
        self.cost_weight = QDoubleSpinBox(); self.cost_weight.setRange(0, 100); self.cost_weight.setValue(settings.supplier_weights.get("cost", 0.2) * 100)
        self.roi_reduction = QDoubleSpinBox(); self.roi_reduction.setRange(0, 100); self.roi_reduction.setSuffix(" %"); self.roi_reduction.setValue(settings.roi_defect_reduction_pct)
        self.max_line_points = QSpinBox(); self.max_line_points.setRange(500, 50000); self.max_line_points.setValue(settings.max_points_line_chart)
        self.pca_sample = QSpinBox(); self.pca_sample.setRange(200, 20000); self.pca_sample.setValue(settings.pca_sample_size)
        self.min_supplier_volume = QSpinBox(); self.min_supplier_volume.setRange(1, 1000); self.min_supplier_volume.setValue(settings.supplier_min_volume_for_ranking)
        self.max_labels = QSpinBox(); self.max_labels.setRange(5, 30); self.max_labels.setValue(settings.max_category_labels)
        self.max_label_len = QSpinBox(); self.max_label_len.setRange(8, 40); self.max_label_len.setValue(settings.max_label_length)
        self.max_xtick_labels = QSpinBox(); self.max_xtick_labels.setRange(4, 15); self.max_xtick_labels.setValue(settings.max_xtick_labels)
        self.dark_mode = QCheckBox("Enable dark mode"); self.dark_mode.setChecked(settings.dark_mode)
        self.alert_defect_rate = QDoubleSpinBox(); self.alert_defect_rate.setRange(0, 100); self.alert_defect_rate.setSuffix(" %"); self.alert_defect_rate.setValue(settings.alert_defect_rate_threshold)
        self.alert_copq_spike = QDoubleSpinBox(); self.alert_copq_spike.setRange(0, 1e9); self.alert_copq_spike.setValue(settings.alert_copq_spike_threshold)
        self.enable_realtime = QCheckBox("Enable real-time monitoring"); self.enable_realtime.setChecked(settings.enable_realtime_monitoring)
        self.realtime_interval = QSpinBox(); self.realtime_interval.setRange(1000, 60000); self.realtime_interval.setSuffix(" ms"); self.realtime_interval.setValue(settings.realtime_interval_ms)

        general_layout.addRow("Supplier weight - defect", self.defect_weight)
        general_layout.addRow("Supplier weight - delivery", self.delivery_weight)
        general_layout.addRow("Supplier weight - cost", self.cost_weight)
        general_layout.addRow("COPQ what-if reduction", self.roi_reduction)
        general_layout.addRow("Max line chart points", self.max_line_points)
        general_layout.addRow("PCA sample size", self.pca_sample)
        general_layout.addRow("Min supplier volume for rank", self.min_supplier_volume)
        general_layout.addRow("Max categories per chart", self.max_labels)
        general_layout.addRow("Max label length", self.max_label_len)
        general_layout.addRow("Max X tick labels", self.max_xtick_labels)
        general_layout.addRow("Theme", self.dark_mode)
        general_layout.addRow("Alert: Defect Rate Threshold %", self.alert_defect_rate)
        general_layout.addRow("Alert: COPQ Spike Threshold", self.alert_copq_spike)
        general_layout.addRow("Real-time monitoring", self.enable_realtime)
        general_layout.addRow("Real-time interval", self.realtime_interval)
        tabs.addTab(general_tab, "General")

        adv_tab = QWidget()
        adv_layout = QFormLayout(adv_tab)
        self.anomaly_contamination = QDoubleSpinBox(); self.anomaly_contamination.setRange(0.01, 0.5); self.anomaly_contamination.setSingleStep(0.01); self.anomaly_contamination.setValue(settings.anomaly_contamination)
        self.anomaly_sample_size = QSpinBox(); self.anomaly_sample_size.setRange(100, 100000); self.anomaly_sample_size.setValue(settings.anomaly_sample_size)
        self.model_test_size = QDoubleSpinBox(); self.model_test_size.setRange(0.1, 0.5); self.model_test_size.setSingleStep(0.05); self.model_test_size.setValue(settings.model_test_size)
        self.random_state = QSpinBox(); self.random_state.setRange(0, 9999); self.random_state.setValue(settings.random_state)
        self.classifier_impute = QComboBox(); self.classifier_impute.addItems(["mean", "median", "most_frequent"]); self.classifier_impute.setCurrentText(settings.classifier_impute_strategy)
        self.forecast_method = QComboBox(); self.forecast_method.addItems(["auto", "linear", "arima"]); self.forecast_method.setCurrentText(settings.forecast_method)
        self.qi_winsorize = QCheckBox("Winsorize quality index"); self.qi_winsorize.setChecked(settings.quality_index_winsorize)
        self.supplier_trend_window = QSpinBox(); self.supplier_trend_window.setRange(2, 12); self.supplier_trend_window.setValue(settings.supplier_trend_window)
        self.control_limits_ddof = QSpinBox(); self.control_limits_ddof.setRange(0, 1); self.control_limits_ddof.setValue(settings.control_limits_ddof)

        adv_layout.addRow("Anomaly contamination", self.anomaly_contamination)
        adv_layout.addRow("Anomaly sample size", self.anomaly_sample_size)
        adv_layout.addRow("Model test size", self.model_test_size)
        adv_layout.addRow("Random seed", self.random_state)
        adv_layout.addRow("Classifier imputation strategy", self.classifier_impute)
        adv_layout.addRow("Forecast method", self.forecast_method)
        adv_layout.addRow("Quality index winsorize", self.qi_winsorize)
        adv_layout.addRow("Supplier trend window (months)", self.supplier_trend_window)
        adv_layout.addRow("Control limits ddof", self.control_limits_ddof)
        tabs.addTab(adv_tab, "Advanced Analytics")

        ui_tab = QWidget()
        ui_layout = QFormLayout(ui_tab)
        self.auto_refresh_ms = QSpinBox(); self.auto_refresh_ms.setRange(100, 5000); self.auto_refresh_ms.setValue(settings.auto_refresh_ms)
        self.chart_padding = QDoubleSpinBox(); self.chart_padding.setRange(0.5, 5.0); self.chart_padding.setValue(settings.chart_padding)
        self.enable_progress = QCheckBox("Show progress dialogs"); self.enable_progress.setChecked(settings.enable_progress_dialogs)
        self.pdf_include_charts = QCheckBox("Include charts in PDF export"); self.pdf_include_charts.setChecked(settings.pdf_include_charts)

        ui_layout.addRow("Auto-refresh delay (ms)", self.auto_refresh_ms)
        ui_layout.addRow("Chart padding", self.chart_padding)
        ui_layout.addRow("Enable progress dialogs", self.enable_progress)
        ui_layout.addRow("PDF include charts", self.pdf_include_charts)
        tabs.addTab(ui_tab, "UI")

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_settings(self) -> AppSettings:
        total = self.defect_weight.value() + self.delivery_weight.value() + self.cost_weight.value()
        weights = (
            {"defect": 0.5, "delivery": 0.3, "cost": 0.2}
            if total == 0
            else {
                "defect": self.defect_weight.value() / total,
                "delivery": self.delivery_weight.value() / total,
                "cost": self.cost_weight.value() / total,
            }
        )
        return AppSettings(
            supplier_weights=weights,
            roi_defect_reduction_pct=float(self.roi_reduction.value()),
            monthly_resample_freq=self._settings.monthly_resample_freq,
            max_points_line_chart=int(self.max_line_points.value()),
            pca_sample_size=int(self.pca_sample.value()),
            enable_sampling=self._settings.enable_sampling,
            dark_mode=self.dark_mode.isChecked(),
            preferred_date_column=self._settings.preferred_date_column,
            confidence_z_score=self._settings.confidence_z_score,
            supplier_min_volume_for_ranking=int(self.min_supplier_volume.value()),
            rolling_window_points=self._settings.rolling_window_points,
            max_category_labels=int(self.max_labels.value()),
            max_label_length=int(self.max_label_len.value()),
            auto_refresh_ms=int(self.auto_refresh_ms.value()),
            chart_padding=float(self.chart_padding.value()),
            max_xtick_labels=int(self.max_xtick_labels.value()),
            alert_defect_rate_threshold=float(self.alert_defect_rate.value()),
            alert_copq_spike_threshold=float(self.alert_copq_spike.value()),
            enable_realtime_monitoring=self.enable_realtime.isChecked(),
            realtime_interval_ms=int(self.realtime_interval.value()),
            anomaly_contamination=float(self.anomaly_contamination.value()),
            model_test_size=float(self.model_test_size.value()),
            random_state=int(self.random_state.value()),
            quality_index_winsorize=self.qi_winsorize.isChecked(),
            quality_index_winsor_limits=(0.05, 0.95),
            supplier_trend_window=int(self.supplier_trend_window.value()),
            forecast_method=self.forecast_method.currentText(),
            anomaly_sample_size=int(self.anomaly_sample_size.value()),
            classifier_impute_strategy=self.classifier_impute.currentText(),
            control_limits_ddof=int(self.control_limits_ddof.value()),
            enable_progress_dialogs=self.enable_progress.isChecked(),
            pdf_include_charts=self.pdf_include_charts.isChecked(),
        )

class FilterPresetManager:
    def __init__(self, path: str = FILTER_PRESETS_FILE):
        self.path = Path(path)

    def load_presets(self) -> Dict[str, FilterState]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            presets = {}
            for name, state_dict in data.items():
                start = None
                end = None
                if state_dict.get('start_date'):
                    start = pd.Timestamp(state_dict['start_date'])
                if state_dict.get('end_date'):
                    end = pd.Timestamp(state_dict['end_date'])
                presets[name] = FilterState(
                    part_type=state_dict.get('part_type', 'All Part Types'),
                    defect_type=state_dict.get('defect_type', 'All Defect Types'),
                    severity=state_dict.get('severity', 'All Severities'),
                    material=state_dict.get('material', 'All Materials'),
                    start_date=start,
                    end_date=end
                )
            return presets
        except Exception as exc:
            logger.exception("Failed to load filter presets: %s", exc)
            return {}

    def save_presets(self, presets: Dict[str, FilterState]) -> None:
        data = {}
        for name, state in presets.items():
            data[name] = {
                'part_type': state.part_type,
                'defect_type': state.defect_type,
                'severity': state.severity,
                'material': state.material,
                'start_date': state.start_date.isoformat() if state.start_date else None,
                'end_date': state.end_date.isoformat() if state.end_date else None,
            }
        try:
            self.path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        except Exception as exc:
            logger.exception("Failed to save filter presets: %s", exc)

class AlertManager:
    def __init__(self, message_center: Optional[MessageCenter], settings: AppSettings):
        self.message_center = message_center
        self.settings = settings
        self.alerts = []
        self.last_emitted: Dict[str, str] = {}

    def _post(self, key: str, message: str) -> None:
        if self.last_emitted.get(key) == message:
            return
        self.last_emitted[key] = message
        if self.message_center is not None:
            self.message_center.post("ALERT", message)

    def check_alerts(self, df: pd.DataFrame):
        if df.empty:
            return

        if COL_DEFECT_RATE in df.columns:
            overall_rate = safe_numeric(df[COL_DEFECT_RATE]).mean() * 100
            if pd.notna(overall_rate) and overall_rate > self.settings.alert_defect_rate_threshold:
                msg = f"High overall defect rate: {overall_rate:.2f}% (threshold {self.settings.alert_defect_rate_threshold}%)"
                self._post("defect_rate", msg)
                self.alerts.append(("defect_rate", datetime.now(), msg))

        if COL_COST_IMPACT in df.columns and COL_DATE in df.columns:
            temp = df[[COL_DATE, COL_COST_IMPACT]].copy()
            temp[COL_DATE] = pd.to_datetime(temp[COL_DATE], errors='coerce')
            temp[COL_COST_IMPACT] = safe_numeric(temp[COL_COST_IMPACT]).fillna(0)
            temp = temp.dropna(subset=[COL_DATE])
            if temp.empty:
                return

            daily = temp.set_index(COL_DATE)[COL_COST_IMPACT].resample("D").sum()
            if len(daily) > 7:
                ma = daily.rolling(7).mean()
                latest = daily.iloc[-1]
                baseline = ma.iloc[-1]
                if pd.notna(baseline) and latest > baseline + self.settings.alert_copq_spike_threshold:
                    msg = f"COPQ spike: today's cost {latest:.0f} vs 7-day avg {baseline:.0f}"
                    self._post("copq_spike", msg)
                    self.alerts.append(("copq_spike", datetime.now(), msg))

# ----------------------------------------------------------------------
# Main Window
# ----------------------------------------------------------------------
class QualityDashboardWindow(QMainWindow):
    NAV_ITEMS = [
        ("Overview", "Management", "⊞", "#2F6BFF"),
        ("Root Cause", "Root Cause", "⚡", "#FF6078"),
        ("Trends", "Trends", "📈", "#1FD0B4"),
        ("Suppliers", "Suppliers", "🏢", "#F5A524"),
        ("COPQ", "COPQ", "💲", "#FF7C8E"),
        ("Quality Index", "Quality Index", "★", "#28C7FA"),
        ("Audit", "Audit Readiness", "✓", "#1FD0B4"),
        ("Advanced", "Advanced Analytics", "⚗", "#8EA3C7"),
        ("Anomalies", "Anomaly Detection", "⚠", "#FF6078"),
        ("Prediction", "Predictive Model", "🎯", "#2F6BFF"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1720, 1060)

        self.settings_manager = SettingsManager()
        self.settings = self.settings_manager.load()
        self.store = QualityDataStore()
        self.analytics = AnalyticsEngine(self.settings)
        self.plot_manager = PlotManager(self.settings)
        self.loader_thread: Optional[DataLoaderThread] = None
        self.current_filtered_df = pd.DataFrame()
        self.alert_manager = None
        self.preset_manager = FilterPresetManager()

        self.real_time_timer = QTimer()
        self.real_time_timer.timeout.connect(self.simulate_new_data)

        self._build_ui()
        self._apply_styles()
        self._apply_branding()
        self._load_filter_presets()
        self.alert_manager = AlertManager(self.message_center, self.settings)

    def _generate_icon(self, symbol: str, color_hex: str) -> QIcon:
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        
        base_color = QColor(color_hex)
        if self.settings.dark_mode:
            bg_color = base_color.darker(300)
            bg_color.setAlpha(150)
        else:
            bg_color = base_color.lighter(180)
            bg_color.setAlpha(150)

        painter.setPen(Qt.NoPen)
        painter.setBrush(bg_color)
        painter.drawEllipse(2, 2, 28, 28)
        
        painter.setPen(base_color)
        font = painter.font()
        font.setPixelSize(18)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignCenter, symbol)
        
        painter.end()
        return QIcon(pixmap)

    def toggle_sidebar(self):
        width = self.sidebar.width()
        target = self.settings.collapsed_sidebar_width if width == self.settings.sidebar_width else self.settings.sidebar_width
        
        self.animation1 = QPropertyAnimation(self.sidebar, b"minimumWidth")
        self.animation1.setDuration(self.settings.animation_duration)
        self.animation1.setStartValue(width)
        self.animation1.setEndValue(target)
        self.animation1.setEasingCurve(QEasingCurve.InOutQuart)
        
        self.animation2 = QPropertyAnimation(self.sidebar, b"maximumWidth")
        self.animation2.setDuration(self.settings.animation_duration)
        self.animation2.setStartValue(width)
        self.animation2.setEndValue(target)
        self.animation2.setEasingCurve(QEasingCurve.InOutQuart)
        
        self.animation1.start()
        self.animation2.start()
        
        self.settings.sidebar_collapsed = not self.settings.sidebar_collapsed
        
        if self.settings.sidebar_collapsed:
            self.brand_name_label.hide()
            self.brand_sub_label.hide()
        else:
            self.brand_name_label.show()
            self.brand_sub_label.show()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(self.settings.sidebar_width)
        side_layout = QVBoxLayout(self.sidebar)
        side_layout.setContentsMargins(18, 18, 18, 18)
        side_layout.setSpacing(12)

        brand_row = QHBoxLayout()
        
        self.toggle_sidebar_btn = QToolButton()
        self.toggle_sidebar_btn.setText("☰")
        self.toggle_sidebar_btn.setObjectName("hamburgerBtn")
        self.toggle_sidebar_btn.clicked.connect(self.toggle_sidebar)
        brand_row.addWidget(self.toggle_sidebar_btn)
        
        self.brand_icon = QLabel()
        self.brand_icon.setFixedSize(42, 42)
        brand_text = QVBoxLayout()
        self.brand_name_label = QLabel(APP_NAME)
        self.brand_name_label.setObjectName("sidebarBrand")
        self.brand_sub_label = QLabel("Quality platform")
        self.brand_sub_label.setObjectName("sidebarBrandSub")
        brand_text.addWidget(self.brand_name_label)
        brand_text.addWidget(self.brand_sub_label)
        brand_row.addWidget(self.brand_icon)
        brand_row.addLayout(brand_text)
        brand_row.addStretch()
        side_layout.addLayout(brand_row)

        self.nav_list = QListWidget()
        self.nav_list.setObjectName("navList")
        self.nav_list.setIconSize(QSize(24, 24))
        for text, _, symbol, color in self.NAV_ITEMS:
            item = QListWidgetItem(text)
            item.setIcon(self._generate_icon(symbol, color))
            self.nav_list.addItem(item)
        self.nav_list.setCurrentRow(0)
        side_layout.addWidget(self.nav_list, 1)

        preset_layout = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Load preset...")
        self.save_preset_btn = QPushButton("Save")
        self.save_preset_btn.setFixedWidth(50)
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addWidget(self.save_preset_btn)
        side_layout.addLayout(preset_layout)

        side_footer = QLabel(f"{APP_NAME} {APP_VERSION}")
        side_footer.setObjectName("sidebarFooter")
        side_layout.addWidget(side_footer)
        root.addWidget(self.sidebar)

        main_shell = QVBoxLayout()
        root.addLayout(main_shell, 1)

        topbar = QFrame()
        topbar.setObjectName("topbar")
        topbar_layout = QHBoxLayout(topbar)
        topbar_layout.setContentsMargins(18, 14, 18, 14)

        self.breadcrumb_label = QLabel("Home / Management")
        self.breadcrumb_label.setObjectName("breadcrumb")
        topbar_layout.addWidget(self.breadcrumb_label)
        topbar_layout.addStretch()

        self.dataset_label = QLabel("No dataset loaded")
        self.dataset_label.setObjectName("topbarSubtitle")
        topbar_layout.addWidget(self.dataset_label)
        topbar_layout.addStretch()

        self.last_refresh_label = QLabel("Last refresh: --")
        self.last_refresh_label.setObjectName("topbarMeta")
        topbar_layout.addWidget(self.last_refresh_label)
        topbar_layout.addStretch()

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search data...")
        self.search_edit.setFixedWidth(200)
        self.search_edit.setClearButtonEnabled(True)
        topbar_layout.addWidget(self.search_edit)

        self.theme_btn = QToolButton()
        self.theme_btn.setText("☀️" if self.settings.dark_mode else "🌙")
        self.theme_btn.setToolTip("Toggle Dark/Light Mode")
        self.theme_btn.clicked.connect(self.toggle_theme)
        topbar_layout.addWidget(self.theme_btn)

        self.notif_btn = QToolButton()
        self.notif_btn.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
        self.notif_btn.setToolTip("Notifications")
        topbar_layout.addWidget(self.notif_btn)

        self.user_btn = QToolButton()
        self.user_btn.setText("👤")
        self.user_btn.setToolTip("User Profile")
        topbar_layout.addWidget(self.user_btn)

        main_shell.addWidget(topbar)

        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(10)
        self.card_rows = StatCard("Rows", "0", "#2F6BFF")
        self.card_defects = StatCard("Defects", "0", "#FF6078")
        self.card_copq = StatCard("COPQ", "0", "#F5A524")
        self.card_qi = StatCard("Quality Index", "0", "#1FD0B4")
        for card in [self.card_rows, self.card_defects, self.card_copq, self.card_qi]:
            kpi_row.addWidget(card)
        main_shell.addLayout(kpi_row)

        command_bar = QHBoxLayout()
        self.load_btn = QPushButton("Load File")
        self.db_btn = QPushButton("SQLite DB")
        self.sql_btn = QPushButton("SQL Server/PG")  
        self.export_pdf_btn = QPushButton("Export PDF")
        self.export_excel_btn = QPushButton("Export Excel")
        self.settings_btn = QPushButton("Settings")
        self.realtime_btn = QPushButton("Real-time")
        self.realtime_btn.setCheckable(True)
        self.realtime_btn.toggled.connect(self.toggle_realtime)

        command_bar.addWidget(self.load_btn)
        command_bar.addWidget(self.db_btn)
        command_bar.addWidget(self.sql_btn)
        command_bar.addWidget(self.export_pdf_btn)
        command_bar.addWidget(self.export_excel_btn)
        command_bar.addWidget(self.settings_btn)
        command_bar.addWidget(self.realtime_btn)
        main_shell.addLayout(command_bar)

        self.filter_bar = QFrame()
        self.filter_bar.setObjectName("filterBar")
        filter_layout = QHBoxLayout(self.filter_bar)
        filter_layout.setContentsMargins(14, 12, 14, 12)
        filter_layout.setSpacing(8)

        self.part_combo = QComboBox()
        self.defect_combo = QComboBox()
        self.severity_combo = QComboBox()
        self.material_combo = QComboBox()
        
        self.date_from = QDateEdit(calendarPopup=True)
        self.date_to = QDateEdit(calendarPopup=True)
        self.date_from.setEnabled(False)
        self.date_to.setEnabled(False)

        filter_layout.addWidget(QLabel("Part"))
        filter_layout.addWidget(self.part_combo)
        filter_layout.addWidget(QLabel("Defect"))
        filter_layout.addWidget(self.defect_combo)
        filter_layout.addWidget(QLabel("Severity"))
        filter_layout.addWidget(self.severity_combo)
        filter_layout.addWidget(QLabel("Material"))
        filter_layout.addWidget(self.material_combo)
        filter_layout.addWidget(QLabel("From"))
        filter_layout.addWidget(self.date_from)
        filter_layout.addWidget(QLabel("To"))
        filter_layout.addWidget(self.date_to)
        self.refresh_btn = QPushButton("Apply")
        filter_layout.addWidget(self.refresh_btn)
        self.filter_toggle_btn = QPushButton("▼")
        self.filter_toggle_btn.setFixedWidth(30)
        self.filter_toggle_btn.clicked.connect(self.toggle_filter_bar)
        filter_layout.addWidget(self.filter_toggle_btn)

        main_shell.addWidget(self.filter_bar)

        self.chips_widget = QWidget()
        chips_layout = QHBoxLayout(self.chips_widget)
        chips_layout.setContentsMargins(0, 0, 0, 0)
        chips_layout.setSpacing(6)
        main_shell.addWidget(self.chips_widget)

        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)
        main_shell.addWidget(split, 1)

        self.stack = QStackedWidget()
        split.addWidget(self.stack)

        right_rail = QWidget()
        right_layout = QVBoxLayout(right_rail)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        quality_surface = Surface("Data Quality Summary")
        self.quality_summary = QTextEdit()
        self.quality_summary.setReadOnly(True)
        self.quality_summary.setMinimumWidth(320)
        self.quality_summary.setFrameShape(QFrame.NoFrame)
        quality_surface.layout.addWidget(self.quality_summary)

        message_surface = Surface("Activity Feed")
        self.message_center = MessageCenter()
        message_surface.layout.addWidget(self.message_center)

        right_layout.addWidget(quality_surface, 3)
        right_layout.addWidget(message_surface, 2)
        split.addWidget(right_rail)
        
        # Globally prioritizing graphs horizontally as well
        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 1)

        self.management_page = ManagementPage(self.plot_manager)
        self.root_cause_page = RootCausePage(self.plot_manager)
        self.trend_page = TrendPage(self.plot_manager, self.settings)
        self.supplier_page = SupplierPage(self.plot_manager)
        self.copq_page = CopqPage(self.plot_manager)
        self.quality_index_page = QualityIndexPage(self.plot_manager, self.settings)
        self.audit_page = AuditPage(self.plot_manager)
        self.advanced_page = AdvancedPage(self.plot_manager, self.settings)
        self.anomaly_page = AnomalyPage(self.plot_manager, self.analytics)
        self.prediction_page = PredictionPage(self.plot_manager, self.analytics)

        self.pages = [
            self.management_page, self.root_cause_page, self.trend_page, self.supplier_page,
            self.copq_page, self.quality_index_page, self.audit_page, self.advanced_page,
            self.anomaly_page, self.prediction_page
        ]
        for page in self.pages:
            self.stack.addWidget(page)

        status = QStatusBar()
        self.status_label = QLabel("Ready")
        status.addWidget(self.status_label)
        self.setStatusBar(status)

        self.nav_list.currentRowChanged.connect(self._change_page)
        self.load_btn.clicked.connect(self.load_data)
        self.db_btn.clicked.connect(self.connect_database)
        self.sql_btn.clicked.connect(self.connect_enterprise_sql)
        self.refresh_btn.clicked.connect(self.refresh_all)
        self.settings_btn.clicked.connect(self.open_settings)
        self.export_pdf_btn.clicked.connect(self.export_pdf_report)
        self.export_excel_btn.clicked.connect(self.export_excel_report)
        self.preset_combo.currentIndexChanged.connect(self.load_preset)
        self.save_preset_btn.clicked.connect(self.save_current_preset)
        self.search_edit.textChanged.connect(self._queue_refresh)
        self.search_edit.returnPressed.connect(self.refresh_all)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._auto_refresh_after_filter_change)

        for combo in [self.part_combo, self.defect_combo, self.severity_combo, self.material_combo]:
            combo.currentIndexChanged.connect(self._queue_refresh)
        self.date_from.dateChanged.connect(self._queue_refresh)
        self.date_to.dateChanged.connect(self._queue_refresh)

        self.filter_controller = FilterController(
            self.store,
            {
                COL_PART_TYPE: self.part_combo,
                COL_DEFECT_TYPE: self.defect_combo,
                COL_DEFECT_SEVERITY: self.severity_combo,
                COL_MATERIAL: self.material_combo,
            },
            self.date_from,
            self.date_to,
        )

    def toggle_theme(self):
        self.settings.dark_mode = not self.settings.dark_mode
        self.theme_btn.setText("☀️" if self.settings.dark_mode else "🌙")
        self.plot_manager.update_palette()
        self._apply_styles()
        
        for index in range(self.nav_list.count()):
            item = self.nav_list.item(index)
            _, _, symbol, color = self.NAV_ITEMS[index]
            item.setIcon(self._generate_icon(symbol, color))

        if not self.store.df.empty:
            self._render_single_page(self.stack.currentIndex())
        self._notify("info", f"Switched to {'Dark' if self.settings.dark_mode else 'Light'} mode.")

    def _apply_branding(self) -> None:
        icon_path = locate_asset(ICON_FILE)
        if icon_path:
            icon = QIcon(icon_path)
            self.setWindowIcon(icon)
            QApplication.instance().setWindowIcon(icon)
            pm = QPixmap(icon_path)
            if not pm.isNull():
                self.brand_icon.setPixmap(pm.scaled(42, 42, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            pm = QPixmap(42, 42)
            pm.fill(QColor("#2F6BFF"))
            self.brand_icon.setPixmap(pm)

    def _apply_styles(self) -> None:
        if self.settings.dark_mode:
            self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #08111F; color: #EAF2FF; font-family: "Segoe UI", Arial, sans-serif; font-size: 12px; }
            QFrame#sidebar { background-color: #0B1425; border: 1px solid #182944; border-radius: 18px; }
            QLabel#sidebarBrand { font-size: 18px; font-weight: 700; color: #F4F8FF; }
            QLabel#sidebarBrandSub, QLabel#sidebarFooter { font-size: 11px; color: #8EA3C7; }
            QListWidget#navList { background: transparent; border: none; outline: none; padding: 4px; }
            QListWidget#navList::item { background: transparent; border: 1px solid transparent; color: #AAC0E3; border-radius: 12px; padding: 12px 14px; margin-bottom: 4px; }
            QListWidget#navList::item:selected { background-color: #13284C; border: 1px solid #2758AF; color: white; }
            QFrame#topbar { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0C1730, stop:0.45 #102042, stop:1 #13294F); border: 1px solid #1E355C; border-radius: 16px; }
            QLabel#topbarTitle { font-size: 22px; font-weight: 700; color: #F4F8FF; }
            QLabel#topbarSubtitle, QLabel#topbarMeta, QLabel#breadcrumb { font-size: 11px; color: #9FB3D6; }
            QFrame#filterBar, QFrame#surface, QFrame#statCard { background-color: #0F1A2E; border: 1px solid #1E355C; border-radius: 16px; }
            QLabel#surfaceTitle { color: #D9E7FF; font-size: 13px; font-weight: 700; padding-bottom: 4px; }
            QLabel#pageTitle { font-size: 20px; font-weight: 700; color: #F3F8FF; }
            QLabel#pageSubtitle { font-size: 11px; color: #96ADD1; margin-bottom: 4px; }
            QLabel#statCardTitle { color: #8EA3C7; font-size: 11px; }
            QLabel#statCardValue { color: #F4F8FF; font-size: 25px; font-weight: 700; }
            QPushButton { background-color: #1745B3; color: white; border: 1px solid #2F6BFF; border-radius: 10px; padding: 8px 14px; font-weight: 600; }
            QPushButton:hover { background-color: #245DE5; }
            QPushButton:pressed { background-color: #123A97; }
            QPushButton:checked { background-color: #186F63; border: 1px solid #1FD0B4; }
            QComboBox, QDateEdit, QTextEdit, QTableView, QListWidget, QLineEdit { background-color: #091427; color: #EAF2FF; border: 1px solid #20375D; border-radius: 10px; padding: 6px 8px; }
            QTableView { gridline-color: #182A47; selection-background-color: #173D8A; alternate-background-color: #0D182E; }
            QHeaderView::section { background-color: #11233D; color: #DDE8FF; border: none; border-bottom: 1px solid #20375D; padding: 8px; font-weight: 600; }
            QToolButton#hamburgerBtn { background: transparent; border: none; color: #EAF2FF; font-size: 18px; font-weight: bold; }
            QToolButton#hamburgerBtn:hover { color: #2F6BFF; }
            """)
        else:
            self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #F4F7FB; color: #1E293B; font-family: "Segoe UI", Arial, sans-serif; font-size: 12px; }
            QFrame#sidebar { background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 18px; }
            QLabel#sidebarBrand { font-size: 18px; font-weight: 700; color: #0F172A; }
            QLabel#sidebarBrandSub, QLabel#sidebarFooter { font-size: 11px; color: #64748B; }
            QListWidget#navList { background: transparent; border: none; outline: none; padding: 4px; }
            QListWidget#navList::item { background: transparent; border: 1px solid transparent; color: #475569; border-radius: 12px; padding: 12px 14px; margin-bottom: 4px; }
            QListWidget#navList::item:selected { background-color: #E0E7FF; border: 1px solid #6366F1; color: #4338CA; }
            QFrame#topbar { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #FFFFFF, stop:1 #F8FAFC); border: 1px solid #E2E8F0; border-radius: 16px; }
            QLabel#topbarTitle { font-size: 22px; font-weight: 700; color: #0F172A; }
            QLabel#topbarSubtitle, QLabel#topbarMeta, QLabel#breadcrumb { font-size: 11px; color: #64748B; }
            QFrame#filterBar, QFrame#surface, QFrame#statCard { background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px; }
            QLabel#surfaceTitle { color: #1E293B; font-size: 13px; font-weight: 700; padding-bottom: 4px; }
            QLabel#pageTitle { font-size: 20px; font-weight: 700; color: #0F172A; }
            QLabel#pageSubtitle { font-size: 11px; color: #64748B; margin-bottom: 4px; }
            QLabel#statCardTitle { color: #64748B; font-size: 11px; }
            QLabel#statCardValue { color: #0F172A; font-size: 25px; font-weight: 700; }
            QPushButton { background-color: #2563EB; color: white; border: 1px solid #1D4ED8; border-radius: 10px; padding: 8px 14px; font-weight: 600; }
            QPushButton:hover { background-color: #3B82F6; }
            QPushButton:pressed { background-color: #1E40AF; }
            QPushButton:checked { background-color: #10B981; border: 1px solid #059669; }
            QComboBox, QDateEdit, QTextEdit, QTableView, QListWidget, QLineEdit { background-color: #F8FAFC; color: #1E293B; border: 1px solid #CBD5E1; border-radius: 10px; padding: 6px 8px; }
            QTableView { gridline-color: #E2E8F0; selection-background-color: #BFDBFE; alternate-background-color: #F1F5F9; }
            QHeaderView::section { background-color: #F8FAFC; color: #1E293B; border: none; border-bottom: 1px solid #CBD5E1; padding: 8px; font-weight: 600; }
            QToolButton#hamburgerBtn { background: transparent; border: none; color: #1E293B; font-size: 18px; font-weight: bold; }
            QToolButton#hamburgerBtn:hover { color: #2563EB; }
            """)

    def _change_page(self, index: int) -> None:
        if index < 0:
            return
        self.stack.setCurrentIndex(index)
        self.breadcrumb_label.setText(f"Home / {self.NAV_ITEMS[index][1]}")
        if not self.current_filtered_df.empty:
            self._render_single_page(index)

    def _render_single_page(self, index: int) -> None:
        filtered = self.current_filtered_df
        try:
            if index == 0: self.management_page.render(filtered, self.analytics)
            elif index == 1: self.root_cause_page.render(filtered, self.analytics)
            elif index == 2: self.trend_page.render(filtered, self.analytics)
            elif index == 3: self.supplier_page.render(filtered, self.analytics)
            elif index == 4: self.copq_page.render(filtered, self.analytics)
            elif index == 5: self.quality_index_page.render(filtered, self.analytics)
            elif index == 6: self.audit_page.render(filtered, self.analytics, self.store.quality_report)
            elif index == 7: self.advanced_page.render(filtered, self.analytics)
            elif index == 8: self.anomaly_page.set_data(filtered)
            elif index == 9: self.prediction_page.set_data(filtered)
        except Exception as e:
            logger.exception("Rendering Error: %s", e)
            self._show_error("Rendering Error", f"Failed to render page: {e}")

    def _notify(self, level: str, message: str) -> None:
        self.message_center.post(level, message)
        self.status_label.setText(message)
        if level.lower() in {"error", "critical"}:
            logger.error(message)
        else:
            logger.info(message)

    def _show_error(self, title: str, message: str) -> None:
        self._notify("error", message)
        QMessageBox.critical(self, title, message)

    def _queue_refresh(self) -> None:
        self.status_label.setText("Filters changed. Refreshing soon…")
        self._refresh_timer.start(self.settings.auto_refresh_ms)

    def _auto_refresh_after_filter_change(self) -> None:
        if not self.store.df.empty:
            self.refresh_all()

    def toggle_filter_bar(self):
        visible = self.filter_bar.isVisible()
        self.filter_bar.setVisible(not visible)
        self.filter_toggle_btn.setText("▲" if not visible else "▼")

    def open_settings(self) -> None:
        dlg = SettingsDialog(self, self.settings)
        if dlg.exec_() == QDialog.Accepted:
            self.settings = dlg.get_settings()
            self.settings_manager.save(self.settings)
            self.analytics.settings = self.settings
            self.plot_manager = PlotManager(self.settings)
            self._apply_styles()
            self._notify("info", "Settings updated.")
            if not self.store.df.empty:
                self.refresh_all()

    def toggle_realtime(self, checked: bool) -> None:
        if checked:
            self.real_time_timer.start(self.settings.realtime_interval_ms)
            self._notify("info", "Real-time monitoring started.")
        else:
            self.real_time_timer.stop()
            self._notify("info", "Real-time monitoring stopped.")

    def simulate_new_data(self) -> None:
        base_df = self.store.df 
        if base_df.empty:
            return

        sample = base_df.sample(1).iloc[0]
        new_row = {}
        for col in base_df.columns:
            if col == COL_DATE:
                new_row[col] = datetime.now() + timedelta(seconds=random.randint(60, 3600))
            elif pd.api.types.is_numeric_dtype(base_df[col]):
                base_val = sample[col] if not pd.isna(sample[col]) else 1
                sigma = 0.05 * abs(base_val) if base_val != 0 else 0.05
                new_row[col] = base_val + random.gauss(0, sigma)
            else:
                new_row[col] = sample[col]

        new_df = pd.DataFrame([new_row])
        self.store._lock.lockForWrite()
        try:
            self.store._normalized_df = pd.concat([self.store._normalized_df, new_df], ignore_index=True)
            self.store._quality_report = DataValidator.build_report(self.store._normalized_df)
            self.store._filter_cache.clear()
        finally:
            self.store._lock.unlock()
            
        self._notify("info", "New real-time data point added.")
        self.refresh_all()

    def connect_enterprise_sql(self) -> None:
        dlg = SQLConnectDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            uri, query = dlg.get_selection()
            if not uri or not query:
                return
            try:
                df = self.store.load_from_sql_uri(uri, query)
                self.filter_controller.populate_from_store()
                self.quality_summary.setPlainText(self.store.quality_report.to_text())
                self.dataset_label.setText(f"Enterprise DB Connected")
                self._notify("info", f"Loaded {len(df):,} rows from Enterprise SQL.")
                for warning in self.store.quality_report.warnings[:6]:
                    self._notify("warning", warning)
                self.refresh_all()
            except Exception as e:
                logger.exception("Enterprise SQL Error: %s", e)
                self._show_error("Database Error", str(e))

    def connect_database(self) -> None:
        dlg = DatabaseDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            db_path, table = dlg.get_selection()
            if not db_path or not table:
                return
            try:
                df = self.store.load_from_sqlite(db_path, table)
                self.filter_controller.populate_from_store()
                self.quality_summary.setPlainText(self.store.quality_report.to_text())
                self.dataset_label.setText(f"Dataset: {Path(db_path).name} / {table}")
                self._notify("info", f"Loaded {len(df):,} rows from {table}.")
                for warning in self.store.quality_report.warnings[:6]:
                    self._notify("warning", warning)
                self.refresh_all()
            except Exception as e:
                logger.exception("SQLite DB Error: %s", e)
                self._show_error("Database Error", str(e))

    def load_data(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select quality dataset", "", "Data files (*.xlsx *.xls *.csv *.txt)")
        if not file_path:
            return

        self.load_btn.setEnabled(False)
        self._notify("info", f"Loading {os.path.basename(file_path)}...")
        self.loader_thread = DataLoaderThread(self.store, file_path, self)
        self.loader_thread.data_loaded.connect(self.on_data_loaded)
        self.loader_thread.error_occurred.connect(self.on_data_load_error)
        self.loader_thread.finished.connect(lambda: self.load_btn.setEnabled(True))
        self.loader_thread.start()

    def on_data_loaded(self, df: pd.DataFrame) -> None:
        self.filter_controller.populate_from_store()
        self.quality_summary.setPlainText(self.store.quality_report.to_text())
        self.dataset_label.setText(f"Dataset: {Path(self.loader_thread.file_path).name}" if self.loader_thread else "Dataset loaded")
        self._notify("info", f"Loaded {len(df):,} rows and {len(df.columns)} columns.")
        for warning in self.store.quality_report.warnings[:6]:
            self._notify("warning", warning)
        self.refresh_all()

    def on_data_load_error(self, message: str) -> None:
        self._show_error("Data Load Error", message)

    def _apply_search_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        search_text = self.search_edit.text().strip()
        if not search_text:
            return df

        searchable = df.copy()
        for col in searchable.columns:
            if pd.api.types.is_datetime64_any_dtype(searchable[col]):
                searchable[col] = pd.to_datetime(searchable[col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
            else:
                searchable[col] = searchable[col].astype(str)

        mask = searchable.apply(
            lambda col: col.str.contains(search_text, case=False, na=False, regex=False)
        ).any(axis=1)
        return df.loc[mask].copy()

    def _get_filtered_df(self) -> pd.DataFrame:
        if self.store.df.empty:
            return pd.DataFrame()
        state = self.filter_controller.current_state()
        filtered = self.store.apply_filters(state, date_column=COL_DATE)
        return self._apply_search_filter(filtered)

    def _update_summary_cards(self, filtered: pd.DataFrame) -> None:
        kpis = self.analytics.executive_kpis(filtered)
        self.card_rows.update_value(f"{int(kpis.get('Rows', 0)):,}")
        self.card_defects.update_value(f"{int(kpis.get('Defects', 0)):,}")
        self.card_copq.update_value(f"{kpis.get('COPQ', 0):,.0f}")
        self.card_qi.update_value(f"{kpis.get('QualityIndex', 0):.1f}")

    def refresh_all(self) -> None:
        if self.store.df.empty:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        try:
            filtered = self._get_filtered_df()
            self.current_filtered_df = filtered

            if filtered.empty:
                self._notify("warning", "No rows match the current filters.")
                self.quality_summary.setPlainText(self.store.quality_report.to_text())
                self._update_summary_cards(pd.DataFrame())
                for page in self.pages:
                    if hasattr(page, "render_empty_page"):
                        page.render_empty_page("No data matches the current filters.")
                return

            self._update_summary_cards(filtered)
            self.alert_manager.check_alerts(filtered)
            self.last_refresh_label.setText(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            current_index = self.stack.currentIndex()
            self._render_single_page(current_index)
            
            search_text = self.search_edit.text().strip()
            if search_text:
                self._notify("info", f"Dashboard refreshed using {len(filtered):,} rows matching filters and search: '{search_text}'.")
            else:
                self._notify("info", f"Dashboard refreshed using {len(filtered):,} filtered rows.")

        except Exception as exc:
            logger.exception("Refresh failed: %s", exc)
            self._show_error("Refresh Error", str(exc))

    def export_pdf_report(self) -> None:
        if not HAS_REPORTLAB:
            QMessageBox.warning(self, "Missing dependency", "reportlab is not installed. Install it with 'pip install reportlab' to enable PDF export.")
            return
        if self.store.df.empty:
            QMessageBox.warning(self, "No Data", "Load data before exporting a report.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save PDF report", "", "PDF Files (*.pdf)")
        if not file_path:
            return
            
        if not file_path.lower().endswith('.pdf'):
            file_path += '.pdf'

        try:
            df = self.current_filtered_df if not self.current_filtered_df.empty else self.store.df
            kpis = self.analytics.executive_kpis(df)
            pareto = self.analytics.defect_pareto(df).head(5)
            what_if = self.analytics.copq_what_if(df, self.settings.roi_defect_reduction_pct)

            doc = SimpleDocTemplate(file_path, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
            story = []

            styles = getSampleStyleSheet()
            title_style = styles['Title']
            heading_style = styles['Heading2']
            normal_style = styles['Normal']

            story.append(Paragraph(f"{APP_NAME} Report", title_style))
            story.append(Spacer(1, 0.2*inch))
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", normal_style))
            story.append(Spacer(1, 0.2*inch))

            story.append(Paragraph("Executive KPIs", heading_style))
            story.append(Spacer(1, 0.1*inch))
            
            kpi_data = [["Metric", "Value"]] + [[k, f"{v:,.2f}"] for k, v in kpis.items()]
            kpi_table = Table(kpi_data, colWidths=[2*inch, 2*inch])
            kpi_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,0), 12),
                ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
            ]))
            story.append(kpi_table)
            story.append(Spacer(1, 0.2*inch))

            if not pareto.empty:
                story.append(Paragraph("Top Defect Drivers", heading_style))
                story.append(Spacer(1, 0.1*inch))
                pareto_data = [["Defect Type", "Count", "Cumulative %"]]
                for defect, row in pareto.iterrows():
                    pareto_data.append([defect, str(int(row['Count'])), f"{row['CumulativePct']:.1f}%"])
                pareto_table = Table(pareto_data, colWidths=[2*inch, 1*inch, 1.5*inch])
                pareto_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.grey),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0,0), (-1,0), 12),
                    ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                    ('GRID', (0,0), (-1,-1), 1, colors.black),
                ]))
                story.append(pareto_table)
                story.append(Spacer(1, 0.2*inch))

            story.append(Paragraph("COPQ What-If", heading_style))
            story.append(Spacer(1, 0.1*inch))
            
            copq_data = [
                ["Scenario Metric", "Value"],
                ["Baseline COPQ", f"{what_if['baseline_copq']:,.0f}"],
                ["Assumed reduction", f"{what_if['reduction_pct']:.1f}%"],
                ["Estimated savings", f"{what_if['estimated_savings']:,.0f}"],
                ["Projected COPQ", f"{what_if['projected_copq']:,.0f}"],
            ]
            copq_table = Table(copq_data, colWidths=[2*inch, 2*inch])
            copq_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.grey),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0,0), (-1,0), 12),
                ('BACKGROUND', (0,1), (-1,-1), colors.beige),
                ('GRID', (0,0), (-1,-1), 1, colors.black),
            ]))
            story.append(copq_table)

            doc.build(story)
            self._notify("info", f"PDF report exported to {file_path}")
            QMessageBox.information(self, "Export complete", f"Report saved to:\n{file_path}")
        except Exception as exc:
            logger.exception("PDF export failed: %s", exc)
            self._show_error("Export Error", str(exc))

    def export_excel_report(self) -> None:
        if self.store.df.empty:
            QMessageBox.warning(self, "No Data", "Load data before exporting.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Excel analysis", "", "Excel Files (*.xlsx)")
        if not file_path:
            return
            
        if not file_path.lower().endswith('.xlsx'):
            file_path += '.xlsx'

        try:
            df = self.current_filtered_df if not self.current_filtered_df.empty else self.store.df
            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                df.head(50000).to_excel(writer, sheet_name="FilteredData", index=False)
                pd.DataFrame([self.analytics.executive_kpis(df)]).to_excel(writer, sheet_name="KPIs", index=False)
                self.analytics.defect_pareto(df).to_excel(writer, sheet_name="DefectPareto")
                self.analytics.supplier_scores(df).to_excel(writer, sheet_name="SupplierScores")
                self.analytics.root_cause_drilldown(df).to_excel(writer, sheet_name="RootCauseDrilldown", index=False)
                self.analytics.monthly_copq(df).to_frame("COPQ").to_excel(writer, sheet_name="MonthlyCOPQ")
            self._notify("info", f"Excel analysis exported to {file_path}")
            QMessageBox.information(self, "Export complete", f"Excel analysis saved to:\n{file_path}")
        except Exception as exc:
            logger.exception("Excel export failed: %s", exc)
            self._show_error("Export Error", str(exc))

    def _load_filter_presets(self) -> None:
        presets = self.preset_manager.load_presets()
        self.preset_combo.clear()
        self.preset_combo.addItem("Load preset...")
        for name in presets.keys():
            self.preset_combo.addItem(name)
        self._presets = presets

    def save_current_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "Save Filter Preset", "Enter preset name:")
        if not ok or not name.strip():
            return
        state = self.filter_controller.current_state()
        self._presets[name.strip()] = state
        self.preset_manager.save_presets(self._presets)
        self._load_filter_presets()
        self._notify("info", f"Filter preset '{name}' saved.")

    def load_preset(self, index: int) -> None:
        if index <= 0:
            return
        name = self.preset_combo.currentText()
        if name in self._presets:
            state = self._presets[name]
            self.part_combo.setCurrentText(state.part_type)
            self.defect_combo.setCurrentText(state.defect_type)
            self.severity_combo.setCurrentText(state.severity)
            self.material_combo.setCurrentText(state.material)
            if state.start_date:
                self.date_from.setDate(QDate(state.start_date.year, state.start_date.month, state.start_date.day))
            if state.end_date:
                self.date_to.setDate(QDate(state.end_date.year, state.end_date.month, state.end_date.day))
            self._notify("info", f"Loaded filter preset '{name}'.")
            self.refresh_all()

# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    font = QFontDatabase.systemFont(QFontDatabase.GeneralFont)
    font.setPointSize(10)
    app.setFont(font)

    pm = QPixmap(400, 200)
    pm.fill(QColor("#08111F")) 
    splash = QSplashScreen(pm)
    
    splash.show()
    splash.showMessage("Starting Aeroanalytics 8.2...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
    QApplication.processEvents()

    window = QualityDashboardWindow()
    window.show()
    splash.finish(window)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()