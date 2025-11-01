from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, QComboBox, QPushButton
from PyQt6.QtCore import pyqtSignal, Qt
import pyqtgraph as pg
import numpy as np
import pandas as pd
from typing import Dict, Any, List

# Placeholder for more advanced plotting libraries if needed
# import matplotlib.pyplot as plt 
# from mpl_toolkits.mplot3d import Axes3D

class FeatureCorrelationView(QWidget):
    """Widget to display a feature correlation matrix."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.heatmap = pg.ImageItem()
        self.plot_widget.addItem(self.heatmap)
        layout.addWidget(self.plot_widget)
        self.plot_widget.setTitle("Feature Correlation Matrix")
        self.plot_widget.getAxis('bottom').setLabel('Features')
        self.plot_widget.getAxis('left').setLabel('Features')

    def update_correlation_matrix(self, dataframe: pd.DataFrame):
        if dataframe.empty:
            self.heatmap.setImage(np.array([]))
            return

        correlation_matrix = dataframe.corr().values
        self.heatmap.setImage(correlation_matrix)
        self.heatmap.setRect(pg.QtCore.QRectF(0, 0, correlation_matrix.shape[1], correlation_matrix.shape[0]))

        # Set axis ticks
        feature_names = list(dataframe.columns)
        ticks_x = [(i, name) for i, name in enumerate(feature_names)]
        ticks_y = [(i, name) for i, name in enumerate(feature_names)]
        self.plot_widget.getAxis('bottom').setTicks([ticks_x])
        self.plot_widget.getAxis('left').setTicks([ticks_y])

        # Adjust view
        self.plot_widget.autoRange()

class FeatureSpaceView(QWidget):
    """Widget to display 2D/3D feature space plots."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("X-axis:"))
        self.x_axis_combo = QComboBox()
        control_layout.addWidget(self.x_axis_combo)
        control_layout.addWidget(QLabel("Y-axis:"))
        self.y_axis_combo = QComboBox()
        control_layout.addWidget(self.y_axis_combo)
        control_layout.addWidget(QLabel("Z-axis (3D):"))
        self.z_axis_combo = QComboBox()
        control_layout.addWidget(self.z_axis_combo)
        self.plot_3d_btn = QPushButton("Plot 3D")
        self.plot_3d_btn.clicked.connect(self._plot_3d_placeholder) # Placeholder for 3D
        control_layout.addWidget(self.plot_3d_btn)
        layout.addLayout(control_layout)

        self.plot_widget = pg.PlotWidget()
        self.scatter_plot = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 200))
        self.plot_widget.addItem(self.scatter_plot)
        layout.addWidget(self.plot_widget)
        self.plot_widget.setTitle("2D Feature Space")
        self.plot_widget.setLabel('left', 'Feature Y')
        self.plot_widget.setLabel('bottom', 'Feature X')

        self.x_axis_combo.currentTextChanged.connect(self._update_plot)
        self.y_axis_combo.currentTextChanged.connect(self._update_plot)
        
        self.feature_data = None

    def _plot_3d_placeholder(self):
        print("3D plotting not yet implemented with PyQtGraph's default ScatterPlotItem. Requires custom GLScatterPlotItem or external library.")
        # For actual 3D, would need to use pyqtgraph.opengl.GLViewWidget and GLScatterPlotItem
        # Or integrate with Matplotlib's 3D capabilities.

    def update_feature_data(self, dataframe: pd.DataFrame):
        self.feature_data = dataframe
        feature_names = ['None'] + list(dataframe.columns)
        self.x_axis_combo.clear()
        self.y_axis_combo.clear()
        self.z_axis_combo.clear()
        self.x_axis_combo.addItems(feature_names)
        self.y_axis_combo.addItems(feature_names)
        self.z_axis_combo.addItems(feature_names)
        
        if not dataframe.empty:
            self.x_axis_combo.setCurrentIndex(1) # Select first feature by default
            if len(feature_names) > 2:
                self.y_axis_combo.setCurrentIndex(2) # Select second feature by default
            self._update_plot()

    def _update_plot(self):
        if self.feature_data is None or self.feature_data.empty:
            self.scatter_plot.setData([], [])
            return

        x_feature = self.x_axis_combo.currentText()
        y_feature = self.y_axis_combo.currentText()

        if x_feature != 'None' and y_feature != 'None':
            x_data = self.feature_data[x_feature].values
            y_data = self.feature_data[y_feature].values
            self.scatter_plot.setData(x=x_data, y=y_data)
            self.plot_widget.setLabel('bottom', x_feature)
            self.plot_widget.setLabel('left', y_feature)
            self.plot_widget.setTitle(f"2D Feature Space: {x_feature} vs {y_feature}")
        else:
            self.scatter_plot.setData([], [])
            self.plot_widget.setTitle("2D Feature Space")
            self.plot_widget.setLabel('left', 'Feature Y')
            self.plot_widget.setLabel('bottom', 'Feature X')

class FeatureDistributionView(QWidget):
    """Widget to display statistical distribution views (e.g., histograms, box plots)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Select Feature:"))
        self.feature_combo = QComboBox()
        control_layout.addWidget(self.feature_combo)
        layout.addLayout(control_layout)

        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        self.plot_widget.setTitle("Feature Distribution")
        self.plot_widget.setLabel('left', 'Frequency')
        self.plot_widget.setLabel('bottom', 'Feature Value')

        self.feature_combo.currentTextChanged.connect(self._update_plot)
        self.feature_data = None

    def update_feature_data(self, dataframe: pd.DataFrame):
        self.feature_data = dataframe
        feature_names = ['None'] + list(dataframe.columns)
        self.feature_combo.clear()
        self.feature_combo.addItems(feature_names)
        if not dataframe.empty:
            self.feature_combo.setCurrentIndex(1) # Select first feature by default
            self._update_plot()

    def _update_plot(self):
        if self.feature_data is None or self.feature_data.empty:
            self.plot_widget.clear()
            return

        selected_feature = self.feature_combo.currentText()
        if selected_feature != 'None' and selected_feature in self.feature_data.columns:
            data = self.feature_data[selected_feature].values
            y, x = np.histogram(data, bins=50)
            self.plot_widget.clear()
            self.plot_widget.plot(x, y, stepMode=True, fillLevel=0, brush=(0,0,255,150))
            self.plot_widget.setTitle(f"Distribution of {selected_feature}")
            self.plot_widget.setLabel('bottom', selected_feature)
        else:
            self.plot_widget.clear()
            self.plot_widget.setTitle("Feature Distribution")
            self.plot_widget.setLabel('left', 'Frequency')
            self.plot_widget.setLabel('bottom', 'Feature Value')


class FeatureVisualizationManager(QWidget):
    """
    Manages various visualizations for extracted features.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.feature_dataframe = pd.DataFrame()

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.correlation_view = FeatureCorrelationView()
        self.tabs.addTab(self.correlation_view, "Correlation Matrix")

        self.feature_space_view = FeatureSpaceView()
        self.tabs.addTab(self.feature_space_view, "Feature Space (2D/3D)")

        self.distribution_view = FeatureDistributionView()
        self.tabs.addTab(self.distribution_view, "Distributions")

        # TODO: Add Time series feature evolution and Feature comparison tools
        # self.time_evolution_view = TimeSeriesFeatureEvolutionView()
        # self.tabs.addTab(self.time_evolution_view, "Time Evolution")

        # self.comparison_tools_view = FeatureComparisonToolsView()
        # self.tabs.addTab(self.comparison_tools_view, "Comparison Tools")

    def update_features(self, dataframe: pd.DataFrame):
        """
        Update all visualization views with new feature data.
        """
        self.feature_dataframe = dataframe
        self.correlation_view.update_correlation_matrix(dataframe)
        self.feature_space_view.update_feature_data(dataframe)
        self.distribution_view.update_feature_data(dataframe)
        # self.time_evolution_view.update_feature_data(dataframe)
        # self.comparison_tools_view.update_feature_data(dataframe)