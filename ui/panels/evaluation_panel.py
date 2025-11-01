from PyQt6.QtWidgets import (
    QVBoxLayout, QLabel, QPushButton, QWidget, QComboBox,
    QGroupBox, QFormLayout, QTextEdit, QHBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import pyqtSignal, Qt
from ui.panels.base_panel import BaseControlPanel

class EvaluationPanel(BaseControlPanel):
    evaluation_requested = pyqtSignal(str) # Emits model ID to evaluate
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)

        # 1. Model Selection for Evaluation
        model_selection_group = QGroupBox("Select Model for Evaluation")
        model_selection_layout = QHBoxLayout(model_selection_group)

        self.model_to_evaluate_combo = QComboBox(self)
        self.model_to_evaluate_combo.addItem("Select a trained model...") # Placeholder
        # In a real app, this would be populated by ModelManager
        model_selection_layout.addWidget(self.model_to_evaluate_combo)

        self.evaluate_button = QPushButton("Run Evaluation", self)
        self.evaluate_button.clicked.connect(self._run_evaluation)
        model_selection_layout.addWidget(self.evaluate_button)
        main_layout.addWidget(model_selection_group)

        # 2. Displaying Key Metrics
        metrics_group = QGroupBox("Key Metrics")
        metrics_layout = QFormLayout(metrics_group)

        self.accuracy_label = QLabel("Accuracy: N/A")
        metrics_layout.addRow("Accuracy:", self.accuracy_label)
        self.precision_label = QLabel("Precision: N/A")
        metrics_layout.addRow("Precision:", self.precision_label)
        self.recall_label = QLabel("Recall: N/A")
        metrics_layout.addRow("Recall:", self.recall_label)
        self.f1_score_label = QLabel("F1-Score: N/A")
        metrics_layout.addRow("F1-Score:", self.f1_score_label)
        self.auc_label = QLabel("AUC: N/A")
        metrics_layout.addRow("AUC:", self.auc_label)
        main_layout.addWidget(metrics_group)

        # 3. Interactive Confusion Matrix Visualization (Placeholder)
        confusion_matrix_group = QGroupBox("Confusion Matrix")
        confusion_matrix_layout = QVBoxLayout(confusion_matrix_group)
        self.confusion_matrix_placeholder = QLabel("Confusion Matrix Visualization (Placeholder)")
        self.confusion_matrix_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        confusion_matrix_layout.addWidget(self.confusion_matrix_placeholder)
        main_layout.addWidget(confusion_matrix_group)

        # 4. ROC Curve Plotting (Placeholder)
        roc_curve_group = QGroupBox("ROC Curve")
        roc_curve_layout = QVBoxLayout(roc_curve_group)
        self.roc_curve_placeholder = QLabel("ROC Curve Plot (Placeholder)")
        self.roc_curve_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        roc_curve_layout.addWidget(self.roc_curve_placeholder)
        main_layout.addWidget(roc_curve_group)

        # 5. Cross-validation Results Summary
        cv_results_group = QGroupBox("Cross-Validation Results")
        cv_results_layout = QVBoxLayout(cv_results_group)
        self.cv_table = QTableWidget(self)
        self.cv_table.setColumnCount(3)
        self.cv_table.setHorizontalHeaderLabels(["Fold", "Accuracy", "F1-Score"])
        self.cv_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        cv_results_layout.addWidget(self.cv_table)
        main_layout.addWidget(cv_results_group)

        # 6. Model Comparison Tools (Placeholder)
        model_comparison_group = QGroupBox("Model Comparison")
        model_comparison_layout = QVBoxLayout(model_comparison_group)
        self.model_comparison_placeholder = QLabel("Model Comparison Tools (Placeholder)")
        self.model_comparison_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        model_comparison_layout.addWidget(self.model_comparison_placeholder)
        main_layout.addWidget(model_comparison_group)
        
        self.setLayout(main_layout)

    def _run_evaluation(self):
        selected_model_id = self.model_to_evaluate_combo.currentText()
        if selected_model_id and selected_model_id != "Select a trained model...":
            print(f"Requesting evaluation for model: {selected_model_id}")
            self.evaluation_requested.emit(selected_model_id)
        else:
            print("Please select a model to evaluate.")

    def on_evaluation_results(self, results: dict):
        """
        Updates the UI with evaluation results.
        Results dict expected to contain:
        - 'accuracy', 'precision', 'recall', 'f1_score', 'auc'
        - 'confusion_matrix': (e.g., list of lists)
        - 'roc_curve': (e.g., dict with 'fpr', 'tpr')
        - 'cross_validation_folds': (e.g., list of dicts with 'fold', 'accuracy', 'f1_score')
        """
        self.accuracy_label.setText(f"Accuracy: {results.get('accuracy', 'N/A'):.4f}")
        self.precision_label.setText(f"Precision: {results.get('precision', 'N/A'):.4f}")
        self.recall_label.setText(f"Recall: {results.get('recall', 'N/A'):.4f}")
        self.f1_score_label.setText(f"F1-Score: {results.get('f1_score', 'N/A'):.4f}")
        self.auc_label.setText(f"AUC: {results.get('auc', 'N/A'):.4f}")

        # Update Confusion Matrix (placeholder for actual visualization)
        if 'confusion_matrix' in results:
            print(f"Confusion Matrix Data: {results['confusion_matrix']}")
            self.confusion_matrix_placeholder.setText(f"Confusion Matrix: {results['confusion_matrix']}")

        # Update ROC Curve (placeholder for actual visualization)
        if 'roc_curve' in results:
            print(f"ROC Curve Data: {results['roc_curve']}")
            self.roc_curve_placeholder.setText(f"ROC Curve: {results['roc_curve']}")

        # Update Cross-Validation Table
        self.cv_table.setRowCount(0) # Clear existing rows
        if 'cross_validation_folds' in results:
            for row, fold_data in enumerate(results['cross_validation_folds']):
                self.cv_table.insertRow(row)
                self.cv_table.setItem(row, 0, QTableWidgetItem(str(fold_data.get('fold', 'N/A'))))
                self.cv_table.setItem(row, 1, QTableWidgetItem(f"{fold_data.get('accuracy', 'N/A'):.4f}"))
                self.cv_table.setItem(row, 2, QTableWidgetItem(f"{fold_data.get('f1_score', 'N/A'):.4f}"))
        
        print("Evaluation results updated.")

    def on_evaluation_error(self, error_message: str):
        print(f"Evaluation Error: {error_message}")
        # Display error in a more prominent way, e.g., a QMessageBox
        self.accuracy_label.setText("Accuracy: Error")
        self.precision_label.setText("Precision: Error")
        self.recall_label.setText("Recall: Error")
        self.f1_score_label.setText("F1-Score: Error")
        self.auc_label.setText("AUC: Error")
        self.confusion_matrix_placeholder.setText(f"Error: {error_message}")
        self.roc_curve_placeholder.setText(f"Error: {error_message}")
        self.cv_table.setRowCount(0) # Clear table on error

    def update_available_models(self, model_ids: list):
        self.model_to_evaluate_combo.clear()
        if not model_ids:
            self.model_to_evaluate_combo.addItem("No trained models available")
            self.evaluate_button.setEnabled(False)
        else:
            self.model_to_evaluate_combo.addItems(model_ids)
            self.evaluate_button.setEnabled(True)