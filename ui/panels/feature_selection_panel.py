from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QComboBox, QSlider, QLabel, QPushButton, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSignal
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Dict, Any, List, Optional

from .base_panel import BaseControlPanel, NumericParameter, SliderParameter, EnumParameter

class FeatureSelectionPanel(BaseControlPanel):
    """
    Panel for feature selection tools, including ranking, correlation-based selection,
    feature subset evaluation, and cross-validation.
    """
    
    features_selected = pyqtSignal(pd.DataFrame) # Emits the selected feature DataFrame
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.feature_data: Optional[pd.DataFrame] = None
        self.target_data: Optional[pd.Series] = None
        self.selected_feature_names: List[str] = []
        self._init_ui()
        
    def _init_ui(self):
        super()._init_ui()
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)

        # Input Data Group
        input_group = self.add_parameter_group("Input Data")
        self.input_info_label = QLabel("Load feature matrix and target labels first.")
        input_group.layout().addWidget(self.input_info_label)

        # Selection Method Group
        method_group = self.add_parameter_group("Selection Method")
        self.method_selector = EnumParameter("selection_method", ["None", "Ranking (ANOVA F-value)", "Ranking (Mutual Info)", "Correlation Threshold", "Variance Threshold", "PCA", "Manual Selection"])
        self.add_parameter(method_group, "Method", self.method_selector)
        self.method_selector.value_changed.connect(self._on_method_changed)

        # Method-specific parameters container
        self.method_params_layout = QVBoxLayout()
        method_group.layout().addLayout(self.method_params_layout)
        self.method_params_widgets: Dict[str, QWidget] = {}
        self._init_method_params()

        # Manual Selection Group
        self.manual_selection_group = self.add_parameter_group("Manual Selection")
        self.manual_selection_group.setVisible(False)
        manual_layout = QVBoxLayout(self.manual_selection_group)
        
        self.available_features_list = QListWidget()
        self.available_features_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        manual_layout.addWidget(QLabel("Available Features:"))
        manual_layout.addWidget(self.available_features_list)
        
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.available_features_list.selectAll)
        manual_layout.addWidget(self.select_all_btn)
        
        self.deselect_all_btn = QPushButton("Deselect All")
        self.deselect_all_btn.clicked.connect(self.available_features_list.clearSelection)
        manual_layout.addWidget(self.deselect_all_btn)

        # Action Buttons
        action_layout = QHBoxLayout()
        self.apply_selection_btn = QPushButton("Apply Selection")
        self.apply_selection_btn.clicked.connect(self.apply_feature_selection)
        action_layout.addWidget(self.apply_selection_btn)
        
        self.reset_btn = QPushButton("Reset Selection")
        self.reset_btn.clicked.connect(self.reset_selection)
        action_layout.addWidget(self.reset_btn)
        self.layout.addLayout(action_layout)

        # Selected Features Display
        self.selected_features_group = self.add_parameter_group("Selected Features")
        self.selected_features_list = QListWidget()
        self.selected_features_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.selected_features_count_label = QLabel("Count: 0")
        self.selected_features_group.layout().addWidget(self.selected_features_count_label)
        self.selected_features_group.layout().addWidget(self.selected_features_list)

        self._on_method_changed(self.method_selector.get_value()) # Initialize visibility

    def _init_method_params(self):
        # Ranking (KBest)
        k_best_param = NumericParameter("k_features", min_val=1, max_val=100, step=1, decimals=0, default=10)
        self.add_parameter_to_layout(self.method_params_layout, "Number of Features (k)", k_best_param, "Ranking (ANOVA F-value)")
        self.add_parameter_to_layout(self.method_params_layout, "Number of Features (k)", k_best_param, "Ranking (Mutual Info)")
        self.method_params_widgets["Ranking (ANOVA F-value)"] = k_best_param
        self.method_params_widgets["Ranking (Mutual Info)"] = k_best_param

        # Correlation Threshold
        corr_threshold_param = SliderParameter("correlation_threshold", min_val=0.0, max_val=1.0, step=0.01, decimals=2, default=0.8)
        self.add_parameter_to_layout(self.method_params_layout, "Threshold", corr_threshold_param, "Correlation Threshold")
        self.method_params_widgets["Correlation Threshold"] = corr_threshold_param

        # Variance Threshold
        var_threshold_param = SliderParameter("variance_threshold", min_val=0.0, max_val=1.0, step=0.001, decimals=3, default=0.01)
        self.add_parameter_to_layout(self.method_params_layout, "Threshold", var_threshold_param, "Variance Threshold")
        self.method_params_widgets["Variance Threshold"] = var_threshold_param

        # PCA
        pca_components_param = NumericParameter("pca_components", min_val=1, max_val=100, step=1, decimals=0, default=3)
        self.add_parameter_to_layout(self.method_params_layout, "N Components", pca_components_param, "PCA")
        self.method_params_widgets["PCA"] = pca_components_param

    def add_parameter_to_layout(self, layout: QVBoxLayout, label: str, widget: QWidget, method_name: str):
        """Helper to add a parameter widget to a layout and associate it with a method."""
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel(label))
        h_layout.addWidget(widget)
        h_layout.setStretch(0, 1)
        h_layout.setStretch(1, 3)
        
        container_widget = QWidget()
        container_widget.setLayout(h_layout)
        container_widget.setObjectName(method_name) # Use object name for identification
        container_widget.setVisible(False) # Hidden by default
        layout.addWidget(container_widget)
        
        # Store the container widget for easy access
        if method_name not in self.method_params_widgets:
            self.method_params_widgets[method_name] = []
        self.method_params_widgets[method_name].append(container_widget)
        
        # Also add the actual parameter widget to self.parameters for value tracking
        if isinstance(widget, BaseControlPanel): # BaseControlPanel is a QWidget
            self.parameters[widget.name] = widget
            widget.value_changed.connect(self._on_parameter_changed)


    def _on_method_changed(self, method: str):
        """Show/hide parameters based on selected method."""
        for param_widgets in self.method_params_widgets.values():
            if isinstance(param_widgets, list): # For parameters added via add_parameter_to_layout
                for widget in param_widgets:
                    widget.setVisible(False)
            else: # For single parameter widgets directly added
                param_widgets.setVisible(False)

        if method in self.method_params_widgets:
            param_widgets = self.method_params_widgets[method]
            if isinstance(param_widgets, list):
                for widget in param_widgets:
                    widget.setVisible(True)
            else:
                param_widgets.setVisible(True)
        
        self.manual_selection_group.setVisible(method == "Manual Selection")
        self.update_manual_selection_list()

    def set_feature_data(self, dataframe: pd.DataFrame, target: Optional[pd.Series] = None):
        """Set the feature matrix and optional target labels."""
        self.feature_data = dataframe
        self.target_data = target
        self.input_info_label.setText(f"Features: {dataframe.shape[1]}, Samples: {dataframe.shape[0]}")
        if target is not None:
            self.input_info_label.setText(self.input_info_label.text() + f", Target: {target.name}")
        
        self.update_manual_selection_list()
        self.reset_selection()

    def update_manual_selection_list(self):
        self.available_features_list.clear()
        if self.feature_data is not None:
            for col in self.feature_data.columns:
                item = QListWidgetItem(col)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
                self.available_features_list.addItem(item)

    def apply_feature_selection(self):
        """Apply the selected feature selection method."""
        if self.feature_data is None or self.feature_data.empty:
            print("No feature data available for selection.")
            return

        selected_method = self.method_selector.get_value()
        X = self.feature_data
        y = self.target_data
        
        selected_features_df = pd.DataFrame()
        self.selected_feature_names = []

        try:
            if selected_method == "None":
                selected_features_df = X
                self.selected_feature_names = list(X.columns)
            elif selected_method == "Ranking (ANOVA F-value)":
                if y is None: raise ValueError("Target data required for ANOVA F-value ranking.")
                k = self.parameters["k_features"].get_value()
                selector = SelectKBest(f_classif, k=min(k, X.shape[1]))
                selector.fit(X, y)
                selected_features_df = X.iloc[:, selector.get_support()]
                self.selected_feature_names = list(selected_features_df.columns)
            elif selected_method == "Ranking (Mutual Info)":
                if y is None: raise ValueError("Target data required for Mutual Info ranking.")
                k = self.parameters["k_features"].get_value()
                selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
                selector.fit(X, y)
                selected_features_df = X.iloc[:, selector.get_support()]
                self.selected_feature_names = list(selected_features_df.columns)
            elif selected_method == "Correlation Threshold":
                threshold = self.parameters["correlation_threshold"].get_value()
                corr_matrix = X.corr().abs()
                upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
                selected_features_df = X.drop(columns=to_drop)
                self.selected_feature_names = list(selected_features_df.columns)
            elif selected_method == "Variance Threshold":
                threshold = self.parameters["variance_threshold"].get_value()
                # Scale data before variance thresholding if not already scaled
                scaler = StandardScaler()
                X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
                variances = X_scaled.var()
                to_drop = variances[variances < threshold].index.tolist()
                selected_features_df = X.drop(columns=to_drop)
                self.selected_feature_names = list(selected_features_df.columns)
            elif selected_method == "PCA":
                n_components = self.parameters["pca_components"].get_value()
                if n_components > X.shape[1]:
                    n_components = X.shape[1]
                pca = PCA(n_components=n_components)
                principal_components = pca.fit_transform(X)
                selected_features_df = pd.DataFrame(principal_components, 
                                                    columns=[f'PC{i+1}' for i in range(n_components)])
                self.selected_feature_names = list(selected_features_df.columns)
            elif selected_method == "Manual Selection":
                self.selected_feature_names = [
                    self.available_features_list.item(i).text()
                    for i in range(self.available_features_list.count())
                    if self.available_features_list.item(i).checkState() == Qt.CheckState.Checked
                ]
                selected_features_df = X[self.selected_feature_names]
            else:
                print(f"Unknown selection method: {selected_method}")
                return

            self._update_selected_features_display(selected_features_df)
            self.features_selected.emit(selected_features_df)

        except ValueError as ve:
            print(f"Selection Error: {ve}")
        except Exception as e:
            print(f"An unexpected error occurred during feature selection: {e}")

    def _update_selected_features_display(self, df: pd.DataFrame):
        self.selected_features_list.clear()
        for feature in df.columns:
            self.selected_features_list.addItem(feature)
        self.selected_features_count_label.setText(f"Count: {len(df.columns)}")

    def reset_selection(self):
        """Resets the feature selection to default (all features if data is loaded)."""
        if self.feature_data is not None:
            self.selected_feature_names = list(self.feature_data.columns)
            self._update_selected_features_display(self.feature_data)
            self.method_selector.set_value("None")
            self.features_selected.emit(self.feature_data)
        else:
            self.selected_feature_names = []
            self.selected_features_list.clear()
            self.selected_features_count_label.setText("Count: 0")
            self.method_selector.set_value("None")
