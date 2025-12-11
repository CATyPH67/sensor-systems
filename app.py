# signal_correlation.py
import sys
import math
import json
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QCheckBox, QLineEdit, QListWidget,
    QFileDialog, QMessageBox, QGroupBox, QTextEdit, QSizePolicy,
    QTabWidget, QTableWidget, QTableWidgetItem, QInputDialog, QSpinBox
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# -----------------------------
# Main application
# -----------------------------
class CorrelationApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Корреляция сигнала — PyQt5")
        self.resize(1100, 720)

        # -------------------------
        # встроенные "заводские" значения (твоё текущее состояние)
        # -------------------------
        self._builtin_symbols = ['Ceftr', 'Cef', 'Cefot', 'Cefur', 'Strep', 'Neo', 'Sulf']
        self._builtin_c_values = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
        self._builtin_valueWater = [
            [0.9, 1.03, 0.9, 0.8, 0.7, 0.8, 1],
            [1.3, 1.46, 1.1, 1, 1, 1.26666666666667, 1.5],
            [1.63, 1.88, 1.56, 1.3, 1.3, 1.66666666666667, 1.8],
            [2.03, 2.31, 1.96, 1.6, 1.7, 2.13333333333333, 2.2],
            [2.39, 2.79, 2.32, 1.9, 2, 2.5, 2.5],
            [2.63, 3.2, 2.87, 2.3, 2.3, 3.06666666666667, 2.72],
        ]
        self._builtin_valueNoWater = [
            [2.4, 1.76, 1.7, 1.8, 2.1, 1.7, 2.3],
            [2.7, 2.19, 2, 2.1, 2.5, 2.16666666666667, 2.8],
            [3, 2.62, 2.3, 2.4, 2.8, 2.56666666666667, 3.1],
            [3.4, 3.07, 2.7, 2.7, 3.1, 3.033, 3.5],
            [3.8, 3.54, 3.1, 3, 3.5, 3.4, 3.8],
            [4.1, 4, 3.7, 3.4, 3.8, 3.96, 4],
        ]

        # активные (редактируемые) данные — по умолчанию заводские копии
        self.symbols = list(self._builtin_symbols)
        self.c_values = list(self._builtin_c_values)
        # deep copy arrays of arrays
        self.valueWater = [list(r) for r in self._builtin_valueWater]
        self.valueNoWater = [list(r) for r in self._builtin_valueNoWater]

        # текущее состояние для расчётов
        self.selected_symbol = self.symbols[0] if self.symbols else ''
        self.with_water = False
        self.selected_points = []  # list of (A, C)
        self.k = None
        self.b = None

        # --- UI: вкладки ---
        main_layout = QVBoxLayout(self)
        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        # вкладка расчётов
        self.calc_tab = QWidget()
        tabs.addTab(self.calc_tab, "Расчёт")

        # вкладка редактора таблиц
        self.editor_tab = QWidget()
        tabs.addTab(self.editor_tab, "Редактор таблиц")

        # --- build calc tab ---
        self._build_calc_tab()

        # --- build editor tab ---
        self._build_editor_tab()

        # --- initial populate and plot ---
        self.populate_points_from_tables()
        self.update_regression_and_plots()

    # --------------------
    # Build calculation tab (left controls + right plots)
    # --------------------
    def _build_calc_tab(self):
        layout = QHBoxLayout(self.calc_tab)

        left = QVBoxLayout()
        right = QVBoxLayout()

        # controls group
        controls = QGroupBox("Управление данными")
        ctrl_layout = QVBoxLayout()

        # symbol selector
        h1 = QHBoxLayout()
        lbl_sym = QLabel("Вещество:")
        self.combo = QComboBox()
        self.combo.addItems(self.symbols)
        self.combo.currentTextChanged.connect(self.on_symbol_change)
        h1.addWidget(lbl_sym)
        h1.addWidget(self.combo)
        ctrl_layout.addLayout(h1)

        # water checkbox
        h2 = QHBoxLayout()
        self.chk_water = QCheckBox("С учётом воды")
        self.chk_water.stateChanged.connect(self.on_water_toggle)
        h2.addWidget(self.chk_water)
        ctrl_layout.addLayout(h2)

        # Load CSV (kept)
        h3 = QHBoxLayout()
        self.btn_load = QPushButton("Загрузить CSV (A,C)")
        self.btn_load.clicked.connect(self.load_csv)
        h3.addWidget(self.btn_load)
        ctrl_layout.addLayout(h3)

        # points list
        ctrl_layout.addWidget(QLabel("Текущие точки (A, C):"))
        self.lst_points = QListWidget()
        ctrl_layout.addWidget(self.lst_points)

        # equation and inverse input
        eq_layout = QHBoxLayout()
        self.lbl_eq = QLabel("Уравнение: A = k·log10(C) + b")
        eq_layout.addWidget(self.lbl_eq)
        ctrl_layout.addLayout(eq_layout)

        inv_layout = QHBoxLayout()
        inv_layout.addWidget(QLabel("Ввести A → найти C:"))
        self.edit_A = QLineEdit()
        self.edit_A.setPlaceholderText("число (например 2.5)")
        self.edit_A.returnPressed.connect(self.on_compute_C)
        inv_layout.addWidget(self.edit_A)
        self.btn_compute = QPushButton("Вычислить C")
        self.btn_compute.clicked.connect(self.on_compute_C)
        inv_layout.addWidget(self.btn_compute)
        ctrl_layout.addLayout(inv_layout)

        self.lbl_result = QLabel("C = ")
        ctrl_layout.addWidget(self.lbl_result)

        # ввод C → вычисление A
        inv_layout2 = QHBoxLayout()
        inv_layout2.addWidget(QLabel("Ввести C → найти A:"))
        self.edit_C = QLineEdit()
        self.edit_C.setPlaceholderText("число (например 0.01)")
        self.edit_C.returnPressed.connect(self.on_compute_A)
        inv_layout2.addWidget(self.edit_C)

        self.btn_compute_A = QPushButton("Вычислить A")
        self.btn_compute_A.clicked.connect(self.on_compute_A)
        inv_layout2.addWidget(self.btn_compute_A)

        ctrl_layout.addLayout(inv_layout2)

        self.lbl_result_A = QLabel("A = ")
        ctrl_layout.addWidget(self.lbl_result_A)


        # Apply editor changes quickly button
        btn_apply_editor = QPushButton("Применить последние изменения из редактора")
        btn_apply_editor.clicked.connect(self.apply_editor_to_calc)
        ctrl_layout.addWidget(btn_apply_editor)

        controls.setLayout(ctrl_layout)
        left.addWidget(controls, stretch=0)

        # info / tips
        info = QGroupBox("Инструкция по использованию")
        info_layout = QVBoxLayout()
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setPlainText(
            "Инструкция:\n"
            "1) Во вкладке 'Редактор таблиц' можно полностью изменить таблицы:\n"
            "   - двойной клик по ячейке — редактировать значение A (тока);\n"
            "   - двойной клик по заголовку столбца — изменить название вещества;\n"
            "   - кнопки 'Добавить/Удалить строку' — изменить набор C (концентраций);\n"
            "   - кнопки 'Добавить/Удалить столбец' — добавить/удалить вещество.\n"
            "2) Нажми 'Сохранить изменения' в редакторе, затем в этой вкладке 'Применить последние изменения из редактора' — данные обновятся для расчёта.\n"
            "3) На вкладке 'Расчёт' выбери вещество и режим (с/без воды). Графики и уравнение обновятся автоматически.\n"
            "4) Введи значение A и нажми 'Вычислить C' — приложение использует найденную регрессию A = k·log10(C) + b и выдаст C.\n"
            "5) Можно импортировать/экспортировать таблицы в JSON на вкладке редактора.\n"
        )
        info_layout.addWidget(info_text)
        info.setLayout(info_layout)
        left.addWidget(info, stretch=1)

        # --- графики справа ---
        # Figure A vs C
        self.fig1 = Figure(figsize=(5, 4))
        self.canvas1 = FigureCanvas(self.fig1)
        self.ax1 = self.fig1.add_subplot(111)
        self.ax1.set_title("A vs C")
        self.ax1.set_xlabel("C")
        self.ax1.set_ylabel("A")
        self.ax1.grid(True)

        # Figure A vs -log10(C)
        self.fig2 = Figure(figsize=(5, 4))
        self.canvas2 = FigureCanvas(self.fig2)
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_title("A vs -log10(C)")
        self.ax2.set_xlabel("-log10(C)")
        self.ax2.set_ylabel("A")
        self.ax2.grid(True)

        right.addWidget(self.canvas1, stretch=1)
        right.addWidget(self.canvas2, stretch=1)

        layout.addLayout(left, 30)
        layout.addLayout(right, 70)

        self.calc_tab.setLayout(layout)

    # --------------------
    # Build editor tab
    # --------------------
    def _build_editor_tab(self):
        layout = QVBoxLayout(self.editor_tab)

        top_row = QHBoxLayout()
        self.tbl_selector = QComboBox()
        self.tbl_selector.addItems(["Без учета воды", "С учетом воды"])
        self.tbl_selector.currentIndexChanged.connect(self.on_editor_table_switch)
        top_row.addWidget(QLabel("Редактируемая таблица:"))
        top_row.addWidget(self.tbl_selector)

        # Buttons to add/remove rows/cols and load/save
        btn_row_add = QPushButton("Добавить строку (новое C)")
        btn_row_add.clicked.connect(self.editor_add_row)
        btn_row_del = QPushButton("Удалить выбранную строку")
        btn_row_del.clicked.connect(self.editor_delete_selected_row)

        btn_col_add = QPushButton("Добавить столбец (вещество)")
        btn_col_add.clicked.connect(self.editor_add_column)
        btn_col_del = QPushButton("Удалить выбранный столбец")
        btn_col_del.clicked.connect(self.editor_delete_selected_column)

        btns_rowcol = QHBoxLayout()
        btns_rowcol.addWidget(btn_row_add)
        btns_rowcol.addWidget(btn_row_del)
        btns_rowcol.addWidget(btn_col_add)
        btns_rowcol.addWidget(btn_col_del)

        # Table widget
        self.table_widget = QTableWidget()
        self.table_widget.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.SelectedClicked | QTableWidget.EditKeyPressed)
        self.table_widget.cellChanged.connect(self._on_table_cell_changed)

        # Buttons for save/apply/reset/export/import
        bottom_row = QHBoxLayout()
        self.btn_save_editor = QPushButton("Сохранить изменения (внутренняя)")
        self.btn_save_editor.clicked.connect(self.editor_save_changes)
        self.btn_apply = QPushButton("Применить к расчёту (Apply)")
        self.btn_apply.clicked.connect(self.apply_editor_to_calc)
        self.btn_reset = QPushButton("Сбросить к заводским значениям")
        self.btn_reset.clicked.connect(self.editor_reset_to_builtin)
        self.btn_export = QPushButton("Экспортировать в JSON")
        self.btn_export.clicked.connect(self.editor_export_json)
        self.btn_import = QPushButton("Импортировать JSON")
        self.btn_import.clicked.connect(self.editor_import_json)

        bottom_row.addWidget(self.btn_save_editor)
        bottom_row.addWidget(self.btn_apply)
        bottom_row.addWidget(self.btn_reset)
        bottom_row.addWidget(self.btn_export)
        bottom_row.addWidget(self.btn_import)

        layout.addLayout(top_row)
        layout.addLayout(btns_rowcol)
        layout.addWidget(self.table_widget, stretch=1)
        layout.addLayout(bottom_row)

        self.editor_tab.setLayout(layout)

        # fill table initially with 'Без учета воды'
        self._load_table_into_widget(which='no_water')

        # guard variable to prevent recursive cellChanged handling while populating
        self._suspend_table_change = False

    # -------------------------
    # Editor helpers
    # -------------------------
    def _load_table_into_widget(self, which='no_water'):
        """Load either 'no_water' or 'with_water' table into QTableWidget for editing."""
        self._suspend_table_change = True
        if which == 'with_water':
            table = self.valueWater
        else:
            table = self.valueNoWater

        rows = len(self.c_values)
        cols = len(self.symbols)
        self.table_widget.clear()
        self.table_widget.setRowCount(rows)
        self.table_widget.setColumnCount(cols)

        # set horizontal headers as symbols
        for j, sym in enumerate(self.symbols):
            it = QTableWidgetItem(str(sym))
            it.setFlags(it.flags() | Qt.ItemIsEditable)
            self.table_widget.setHorizontalHeaderItem(j, it)

        # fill rows: show C as leftmost (we'll show C in vertical header)
        for i, c_val in enumerate(self.c_values):
            # vertical header = C value
            self.table_widget.setVerticalHeaderItem(i, QTableWidgetItem(str(c_val)))
            for j in range(cols):
                val = ''
                try:
                    val = table[i][j]
                except Exception:
                    val = ''
                item = QTableWidgetItem("" if val is None else str(val))
                self.table_widget.setItem(i, j, item)

        self._suspend_table_change = False

    def on_compute_A(self):
        """Вычисление A по введённому C."""
        try:
            C = float(self.edit_C.text().replace(",", "."))
            if C <= 0:
                raise ValueError
        except:
            QMessageBox.warning(self, "Ошибка", "Введите корректное значение C (> 0)")
            return

        if self.k is None or self.b is None:
            QMessageBox.warning(self, "Ошибка", "Сначала выберите вещество — нет регрессии.")
            return

        A = self.k * math.log10(C) + self.b
        self.lbl_result_A.setText(f"A = {A:.6f}")


    def on_editor_table_switch(self, idx):
        which = 'no_water' if idx == 0 else 'with_water'
        self._load_table_into_widget(which=which)

    def _on_table_cell_changed(self, row, col):
        # ignore while populating programmatically
        if getattr(self, '_suspend_table_change', False):
            return
        # if header was edited via double click on header, that goes through different path;
        # here handle cell changes (A values)
        # nothing to do instantly — saved on Save button or Apply
        pass

    def editor_add_row(self):
        # prompt for new C value
        text, ok = QInputDialog.getText(self, "Добавить строку", "Введите значение C (например 0.0005):", text="")
        if not ok:
            return
        try:
            c_val = float(text.replace(',', '.'))
        except Exception:
            QMessageBox.warning(self, "Ошибка", "Неправильное значение C.")
            return
        # add to c_values and add default zeros to both tables
        insert_at = len(self.c_values)
        self.c_values.append(c_val)
        for tbl in (self.valueWater, self.valueNoWater):
            # ensure row length = current columns
            new_row = [0.0 for _ in range(len(self.symbols))]
            tbl.append(new_row)
        # reload editor view for current table
        self._load_table_into_widget(which='with_water' if self.tbl_selector.currentIndex() else 'no_water')

    def editor_delete_selected_row(self):
        row = self.table_widget.currentRow()
        if row < 0:
            QMessageBox.information(self, "Удаление строки", "Выберите строку для удаления (клик по строке).")
            return
        confirm = QMessageBox.question(self, "Удалить строку", f"Удалить строку с C={self.c_values[row]}?", QMessageBox.Yes | QMessageBox.No)
        if confirm != QMessageBox.Yes:
            return
        # remove row
        self.c_values.pop(row)
        # remove corresponding row in both tables
        for tbl in (self.valueWater, self.valueNoWater):
            if row < len(tbl):
                tbl.pop(row)
        self._load_table_into_widget(which='with_water' if self.tbl_selector.currentIndex() else 'no_water')

    def editor_add_column(self):
        # ask for column name
        text, ok = QInputDialog.getText(self, "Добавить столбец", "Введите имя вещества (например New):", text="New")
        if not ok:
            return
        name = text.strip() or f"col{len(self.symbols)+1}"
        # append symbol
        self.symbols.append(name)
        # add default values to each row in both tables
        for tbl in (self.valueWater, self.valueNoWater):
            # if table has shorter rows, extend; otherwise append
            for i in range(len(self.c_values)):
                if i < len(tbl):
                    tbl[i].append(0.0)
                else:
                    # row missing for some reason
                    tbl.append([0.0 for _ in range(len(self.symbols))])
        # reload editor
        self._load_table_into_widget(which='with_water' if self.tbl_selector.currentIndex() else 'no_water')
        # update combo in calc tab
        self._refresh_symbol_combo()

    def editor_delete_selected_column(self):
        col = self.table_widget.currentColumn()
        if col < 0:
            QMessageBox.information(self, "Удаление столбца", "Выберите столбец (клик по заголовку или ячейке).")
            return
        confirm = QMessageBox.question(self, "Удалить столбец", f"Удалить столбец '{self.symbols[col]}' ?", QMessageBox.Yes | QMessageBox.No)
        if confirm != QMessageBox.Yes:
            return
        # remove symbol and remove column entries in tables
        self.symbols.pop(col)
        for tbl in (self.valueWater, self.valueNoWater):
            for row in tbl:
                if col < len(row):
                    row.pop(col)
        self._load_table_into_widget(which='with_water' if self.tbl_selector.currentIndex() else 'no_water')
        self._refresh_symbol_combo()

    def editor_save_changes(self):
        """Save changes from the editor widget into the internal arrays (but do not auto-apply to calc)."""
        # read headers (symbols)
        headers = []
        for j in range(self.table_widget.columnCount()):
            it = self.table_widget.horizontalHeaderItem(j)
            name = it.text() if it is not None else f"col{j+1}"
            headers.append(str(name))
        # read vertical headers as c_values
        cvals = []
        for i in range(self.table_widget.rowCount()):
            vhi = self.table_widget.verticalHeaderItem(i)
            if vhi is not None and vhi.text() != '':
                try:
                    cv = float(vhi.text().replace(',', '.'))
                except Exception:
                    # if vertical header isn't numeric, check if first column holds C? We'll keep previous c_values
                    cv = self.c_values[i] if i < len(self.c_values) else 0.0
            else:
                # try to read from first column cell as fallback
                item0 = self.table_widget.item(i, 0)
                if item0:
                    try:
                        cv = float(item0.text().replace(',', '.'))
                    except Exception:
                        cv = self.c_values[i] if i < len(self.c_values) else 0.0
                else:
                    cv = self.c_values[i] if i < len(self.c_values) else 0.0
            cvals.append(float(cv))

        # read table values
        rows = self.table_widget.rowCount()
        cols = self.table_widget.columnCount()
        tbl_vals = []
        for i in range(rows):
            row_vals = []
            for j in range(cols):
                item = self.table_widget.item(i, j)
                if item is None:
                    row_vals.append(0.0)
                else:
                    text = item.text().strip()
                    try:
                        row_vals.append(float(text.replace(',', '.')))
                    except Exception:
                        # leave as 0 if cannot parse
                        row_vals.append(0.0)
            tbl_vals.append(row_vals)

        # commit to internal structures depending on selected editor table
        # We update symbols and c_values globally
        self.symbols = headers
        self.c_values = cvals
        # ensure both tables have the correct shape: rows x cols
        rows = len(self.c_values)
        cols = len(self.symbols)
        def normalize(tbl):
            # ensure number of rows
            while len(tbl) < rows:
                tbl.append([0.0 for _ in range(cols)])
            while len(tbl) > rows:
                tbl.pop()
            # ensure each row length
            for r in tbl:
                while len(r) < cols:
                    r.append(0.0)
                while len(r) > cols:
                    r.pop()
        # replace edited table only; but ensure other table has consistent shape
        if self.tbl_selector.currentIndex() == 0:
            # saved 'no water' table
            self.valueNoWater = [list(r) for r in tbl_vals]
            normalize(self.valueNoWater)
            normalize(self.valueWater)
        else:
            # saved 'with water' table
            self.valueWater = [list(r) for r in tbl_vals]
            normalize(self.valueWater)
            normalize(self.valueNoWater)

        QMessageBox.information(self, "Сохранено", "Изменения сохранены во внутренние данные. Чтобы использовать их в расчётах, нажмите 'Применить к расчёту'.")

        # update calc combobox labels (but do not change selection)
        self._refresh_symbol_combo()

    def editor_reset_to_builtin(self):
        confirm = QMessageBox.question(self, "Сброс", "Вернуть заводские значения (все изменения будут потеряны)?", QMessageBox.Yes | QMessageBox.No)
        if confirm != QMessageBox.Yes:
            return
        # reset active data to builtin copies
        self.symbols = list(self._builtin_symbols)
        self.c_values = list(self._builtin_c_values)
        self.valueWater = [list(r) for r in self._builtin_valueWater]
        self.valueNoWater = [list(r) for r in self._builtin_valueNoWater]
        # reload editor and calc
        self._load_table_into_widget(which='with_water' if self.tbl_selector.currentIndex() else 'no_water')
        self._refresh_symbol_combo()
        QMessageBox.information(self, "Сброшено", "Данные восстановлены к заводским значениям.")

    def editor_export_json(self):
        fname, _ = QFileDialog.getSaveFileName(self, "Сохранить JSON", "tables_export.json", "JSON Files (*.json);;All files (*)")
        if not fname:
            return
        out = {
            "symbols": self.symbols,
            "c_values": self.c_values,
            "valueWater": self.valueWater,
            "valueNoWater": self.valueNoWater
        }
        try:
            with open(fname, 'w', encoding='utf-8') as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "Экспорт", f"Таблицы экспортированы в {fname}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка экспорта", str(e))

    def editor_import_json(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Открыть JSON", "", "JSON Files (*.json);;All files (*)")
        if not fname:
            return
        try:
            with open(fname, 'r', encoding='utf-8') as f:
                obj = json.load(f)
            # basic validation
            if not all(k in obj for k in ("symbols", "c_values", "valueWater", "valueNoWater")):
                QMessageBox.warning(self, "Ошибка", "JSON должен содержать keys: symbols, c_values, valueWater, valueNoWater")
                return
            self.symbols = list(obj["symbols"])
            self.c_values = [float(x) for x in obj["c_values"]]
            self.valueWater = [list(map(float, row)) for row in obj["valueWater"]]
            self.valueNoWater = [list(map(float, row)) for row in obj["valueNoWater"]]
            # reload widget
            self._load_table_into_widget(which='with_water' if self.tbl_selector.currentIndex() else 'no_water')
            self._refresh_symbol_combo()
            QMessageBox.information(self, "Импорт", "JSON успешно импортирован.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка импорта", str(e))

    def _refresh_symbol_combo(self):
        # refresh combo options in calc tab while preserving selection if possible
        current = self.combo.currentText()
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItems(self.symbols)
        # restore selection
        if current in self.symbols:
            self.combo.setCurrentText(current)
            self.selected_symbol = current
        else:
            if self.symbols:
                self.combo.setCurrentIndex(0)
                self.selected_symbol = self.symbols[0]
        self.combo.blockSignals(False)
        # repopulate points and plots, since symbols changed
        self.populate_points_from_tables()
        self.update_regression_and_plots()

    # -------------------------
    # Apply editor changes to calculation (apply last saved internal data)
    # -------------------------
    def apply_editor_to_calc(self):
        # this simply re-populates the calculation data from (possibly modified) internal arrays
        # and redraws everything
        # ensure selected_symbol exists
        if self.selected_symbol not in self.symbols:
            if self.symbols:
                self.selected_symbol = self.symbols[0]
        # update combobox list
        self._refresh_symbol_combo()
        self.populate_points_from_tables()
        self.update_regression_and_plots()
        QMessageBox.information(self, "Применено", "Изменения применены к вкладке расчёта.")

    # -------------------------
    # data & UI handlers for calc
    # -------------------------
    def populate_points_from_tables(self):
        """Fill selected_points using current symbol and with_water flag."""
        self.selected_points = []
        if self.selected_symbol not in self.symbols:
            if self.symbols:
                self.selected_symbol = self.symbols[0]
            else:
                return
        idx = self.symbols.index(self.selected_symbol)
        table = self.valueWater if self.with_water else self.valueNoWater
        # safe handling: if table smaller, pad
        for i, c in enumerate(self.c_values):
            a = 0.0
            try:
                a = table[i][idx]
            except Exception:
                a = 0.0
            try:
                self.selected_points.append((float(a), float(c)))
            except Exception:
                # skip non-numeric rows
                pass
        self.refresh_point_list()

    def refresh_point_list(self):
        self.lst_points.clear()
        for a, c in self.selected_points:
            self.lst_points.addItem(f"A={a}    C={c}")

    def on_symbol_change(self, txt):
        self.selected_symbol = txt
        self.populate_points_from_tables()
        self.update_regression_and_plots()

    def on_water_toggle(self, state):
        self.with_water = bool(state == Qt.Checked)
        self.populate_points_from_tables()
        self.update_regression_and_plots()

    def load_csv(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Выбрать CSV файл", "", "CSV Files (*.csv);;All Files (*)")
        if not fname:
            return
        try:
            pts = []
            with open(fname, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if ',' in line:
                        parts = [p.strip() for p in line.split(',') if p.strip() != '']
                    else:
                        parts = [p.strip() for p in line.split() if p.strip() != '']
                    if len(parts) < 2:
                        continue
                    a = float(parts[0])
                    c = float(parts[1])
                    pts.append((a, c))
            if len(pts) < 2:
                QMessageBox.warning(self, "Ошибка", "В файле должно быть как минимум 2 пары A,C.")
                return
            self.selected_points = pts
            self.refresh_point_list()
            self.update_regression_and_plots()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка чтения файла", str(e))

    # ---------- math: regression and inverse ----------
    def compute_regression(self):
        """Compute linear regression A = k * log10(C) + b using numpy.polyfit"""
        if not self.selected_points or len(self.selected_points) < 2:
            self.k = None
            self.b = None
            return
        A = np.array([p[0] for p in self.selected_points], dtype=float)
        C = np.array([p[1] for p in self.selected_points], dtype=float)
        # ignore non-positive C
        mask = C > 0
        if mask.sum() < 2:
            self.k = None
            self.b = None
            return
        logC = np.log10(C[mask])
        A_masked = A[mask]
        # linear fit: A = k * logC + b
        k, b = np.polyfit(logC, A_masked, 1)
        self.k = float(k)
        self.b = float(b)

    def predict_A_from_C(self, Cs):
        """Given array-like C, return predicted A using k,b (or None)."""
        if self.k is None:
            return None
        Cs = np.array(Cs, dtype=float)
        logC = np.log10(Cs)
        return self.k * logC + self.b

    def predict_C_from_A(self, A_value):
        """Inverse: given A, return C = 10^((A-b)/k). Returns None on fail."""
        if self.k is None or self.k == 0:
            return None
        try:
            logC = (A_value - self.b) / self.k
            C = 10 ** (logC)
            return C
        except Exception:
            return None

    # ---------- plotting ----------
    def update_regression_and_plots(self):
        self.compute_regression()
        # update equation label
        if self.k is None:
            self.lbl_eq.setText("Уравнение: нет данных/ошибка регрессии")
        else:
            self.lbl_eq.setText(f"A = {self.k:.6f}·log10(C) + {self.b:.6f}")

        # plot A vs C (scatter + fitted curve)
        self.ax1.clear()
        self.ax1.set_title("A vs C")
        self.ax1.set_xlabel("C")
        self.ax1.set_ylabel("A")
        self.ax1.grid(True)
        if self.selected_points:
            A = np.array([p[0] for p in self.selected_points], dtype=float)
            C = np.array([p[1] for p in self.selected_points], dtype=float)
            # scatter
            self.ax1.scatter(C, A, label="данные", zorder=3)
            # fitted curve if possible: plot A_pred over C range
            if self.k is not None:
                # make C axis log-spaced for clarity
                Cmin = max(C.min() / 2, 1e-12)
                Cmax = C.max() * 2
                Cs = np.logspace(math.log10(Cmin), math.log10(Cmax), 200)
                A_pred = self.predict_A_from_C(Cs)
                self.ax1.plot(Cs, A_pred, label="регрессия", linewidth=2)
                self.ax1.set_xscale('log')
            self.ax1.legend()

        self.canvas1.draw()

        # plot A vs -log10(C)
        self.ax2.clear()
        self.ax2.set_title("A vs -log10(C)")
        self.ax2.set_xlabel("-log10(C)")
        self.ax2.set_ylabel("A")
        self.ax2.grid(True)
        if self.selected_points:
            A = np.array([p[0] for p in self.selected_points], dtype=float)
            C = np.array([p[1] for p in self.selected_points], dtype=float)
            valid = C > 0
            x = -np.log10(C[valid])
            y = A[valid]
            self.ax2.scatter(x, y, label="данные", zorder=3)
            if self.k is not None:
                # line eq: since A = k*log10(C) + b, and x = -log10(C), we have A = -k*x + b
                x_line = np.linspace(x.min() * 1.2, x.max() * 1.2, 200)
                y_line = -self.k * x_line + self.b
                self.ax2.plot(x_line, y_line, label="линейная регрессия", linewidth=2)
            self.ax2.legend()
        self.canvas2.draw()

    # ---------- UI actions ----------
    def on_compute_C(self):
        text = self.edit_A.text().strip()
        if not text:
            self.lbl_result.setText("C = ")
            return
        try:
            A_val = float(text.replace(',', '.'))
        except ValueError:
            QMessageBox.warning(self, "Ошибка ввода", "Неправильное число A.")
            return
        C = self.predict_C_from_A(A_val)
        if C is None or not np.isfinite(C) or C <= 0:
            self.lbl_result.setText("C = (не вычислено)")
            QMessageBox.information(self, "Результат", "Невозможно вычислить C (проверьте данные и регрессию).")
            return
        self.lbl_result.setText(f"C = {C:.8g}")

# -------------------------
# run
# -------------------------
def main():
    app = QApplication(sys.argv)
    w = CorrelationApp()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
