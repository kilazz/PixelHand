# app/ui/panels/performance.py
import logging
import multiprocessing

try:
    import onnxruntime
except ImportError:
    onnxruntime = None

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QSpinBox,
)

from app.domain.data_models import AppSettings
from app.infrastructure.settings import SettingsManager
from app.shared.constants import (
    DEEP_LEARNING_AVAILABLE,
    DEFAULT_SEARCH_PRECISION,
    SEARCH_PRECISION_PRESETS,
    QuantizationMode,
)

app_logger = logging.getLogger("PixelHand.ui.panels.performance")


class PerformancePanel(QGroupBox):
    log_message = Signal(str, str)
    device_changed = Signal(bool)

    def __init__(self, settings_manager: SettingsManager):
        super().__init__("Performance and AI Model")
        self.settings_manager = settings_manager
        self._init_ui()
        self._detect_and_setup_devices()
        self._connect_signals()
        self.load_settings(settings_manager.settings)

    def _init_ui(self):
        layout = QFormLayout(self)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.device_combo = QComboBox()
        layout.addRow("Device:", self.device_combo)
        self.quant_combo = QComboBox()
        self.quant_combo.addItems([q.value for q in QuantizationMode])
        layout.addRow("Model Precision:", self.quant_combo)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(16, 4096)
        self.batch_size_spin.setSingleStep(16)
        layout.addRow("Batch Size:", self.batch_size_spin)
        self.search_precision_combo = QComboBox()
        layout.addRow("Search Precision:", self.search_precision_combo)
        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setRange(1, (multiprocessing.cpu_count() or 1) * 4)
        layout.addRow("Preprocessing Workers:", self.num_workers_spin)

    def _connect_signals(self):
        self.device_combo.activated.connect(self._on_device_selection_changed)
        self.quant_combo.currentTextChanged.connect(self.settings_manager.set_quantization_mode)
        self.batch_size_spin.valueChanged.connect(self.settings_manager.set_batch_size)
        self.search_precision_combo.currentTextChanged.connect(self.settings_manager.set_search_precision)
        self.num_workers_spin.valueChanged.connect(self.settings_manager.set_num_workers)

    def _detect_and_setup_devices(self):
        self.device_combo.clear()
        if not DEEP_LEARNING_AVAILABLE or onnxruntime is None:
            self.device_combo.addItem("CPUExecutionProvider", "CPUExecutionProvider")
            return
        try:
            providers = onnxruntime.get_available_providers()
            if "CPUExecutionProvider" in providers:
                self.device_combo.addItem("CPUExecutionProvider", "CPUExecutionProvider")
            for pid in providers:
                if pid != "CPUExecutionProvider":
                    self.device_combo.addItem(pid, pid)
        except Exception as e:
            self.log_message.emit(f"Error detecting ONNX providers: {e}", "error")
            if self.device_combo.count() == 0:
                self.device_combo.addItem("CPUExecutionProvider", "CPUExecutionProvider")
        self._on_device_change(self.device_combo.currentData())

    @Slot(int)
    def _on_device_selection_changed(self, index: int):
        if pid := self.device_combo.itemData(index):
            self.settings_manager.set_device(pid)
            self._on_device_change(pid)

    @Slot(str)
    def _on_device_change(self, provider_id: str):
        self.device_changed.emit(provider_id == "CPUExecutionProvider")

    @Slot(str)
    def update_precision_presets(self, scan_mode_name: str):
        self.search_precision_combo.blockSignals(True)
        self.search_precision_combo.clear()
        presets = list(SEARCH_PRECISION_PRESETS.keys())
        self.search_precision_combo.addItems(presets)
        current = self.settings_manager.settings.performance.search_precision
        self.search_precision_combo.setCurrentText(current if current in presets else DEFAULT_SEARCH_PRECISION)
        self.search_precision_combo.blockSignals(False)

    def load_settings(self, s: AppSettings):
        self._detect_and_setup_devices()
        idx = self.device_combo.findData(s.performance.device)
        self.device_combo.setCurrentIndex(idx if idx != -1 else 0)
        self.quant_combo.setCurrentText(s.performance.quantization_mode)
        self.batch_size_spin.setValue(int(s.performance.batch_size))
        self.search_precision_combo.setCurrentText(s.performance.search_precision)
        self.num_workers_spin.setValue(int(s.performance.num_workers))
        self._on_device_change(self.device_combo.currentData())

    def get_selected_quantization(self) -> QuantizationMode:
        return next((q for q in QuantizationMode if q.value == self.quant_combo.currentText()), QuantizationMode.FP16)
