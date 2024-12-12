import json
import sys
import uuid
from datetime import datetime

from PySide6.QtCore import QTimer, QUrl, Slot
from PySide6.QtWebSockets import QWebSocket
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class AgentWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.setMinimumSize(300, 400)

        layout = QVBoxLayout(self)

        # Agent type selector
        self.agent_type = QComboBox()
        self.agent_type.addItem("web_search")
        layout.addWidget(self.agent_type)

        # Close button
        close_button = QPushButton("×")
        close_button.setMaximumWidth(20)
        close_button.clicked.connect(self.close_agent)

        # Header layout
        header = QHBoxLayout()
        header.addWidget(self.agent_type)
        header.addWidget(close_button)
        layout.addLayout(header)

        # Timer interval selector
        self.timer_interval = QComboBox()
        self.timer_interval.addItems(
            ["1 hour", "2 hours", "6 hours", "12 hours", "24 hours"]
        )

        layout.addWidget(self.timer_interval)

        # Setup timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_timer_timeout)

        # Search input
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Enter your query...")
        layout.addWidget(self.query_input)

        # Search button
        self.enable_button = QPushButton("Enable")
        self.enable_button.clicked.connect(self.toggle_action)
        layout.addWidget(self.enable_button)

        # Results display
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        layout.addWidget(self.results_display)

        # WebSocket setup
        self.websocket = QWebSocket()
        self.websocket.connected.connect(self.on_connected)
        self.websocket.textMessageReceived.connect(self.on_message)
        self.websocket.error.connect(self.on_error)

        # Task tracking
        self.current_task_id = None

    @Slot()
    def close_agent(self):
        self.websocket.close()
        self.deleteLater()

    def get_interval_ms(self):
        intervals = {
            "1 hour": 3600000,
            "2 hours": 7200000,
            "6 hours": 21600000,
            "12 hours": 43200000,
            "24 hours": 86400000,
        }
        return intervals[self.timer_interval.currentText()]

    def start_action(self):
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        self.results_display.append(f"Action started at: {formatted_time}")

        self.current_task_id = str(uuid.uuid4())
        ws_url = QUrl(f"ws://localhost:8000/ws/{self.current_task_id}")
        self.websocket.open(ws_url)
        self.enable_button.setEnabled(False)
        self.results_display.append("Connecting...")

    @Slot()
    def toggle_action(self):
        if self.timer.isActive():
            self.timer.stop()
            self.enable_button.setText("Enable")
            self.timer_interval.setEnabled(True)
            self.results_display.clear()
        else:
            self.current_interval = self.get_interval_ms()
            self.start_action()
            if self.current_interval > 0:
                self.timer.start(self.current_interval)
                self.enable_button.setText("Disable")

    @Slot()
    def on_timer_timeout(self):
        self.start_action()
        if self.current_interval > 0:
            self.timer.start(self.current_interval)

    @Slot()
    def on_connected(self):
        query_data = {
            "agent_type": self.agent_type.currentText(),
            "query": self.query_input.text(),
        }
        self.websocket.sendTextMessage(json.dumps(query_data))
        self.results_display.append("Searching...")

    @Slot(str)
    def on_message(self, message):
        data = json.loads(message)

        if data["status"] == "completed":
            self.results_display.append(f"Search Results:\n{data["result"]}")
        elif data["status"] == "failed":
            self.results_display.append(f"Error: {data['error']}")

        self.enable_button.setEnabled(True)
        self.websocket.close()

    @Slot(object)
    def on_error(self, error_code):
        self.results_display.append(f"WebSocket Error: {error_code}")
        self.enable_button.setEnabled(True)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Agent Interface")
        self.setMinimumSize(1000, 800)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Vertical layout for toolbar and grid
        main_layout = QVBoxLayout(main_widget)

        # Toolbar
        toolbar = QHBoxLayout()
        add_agent_btn = QPushButton("Add Agent")
        add_agent_btn.clicked.connect(self.add_agent)
        toolbar.addWidget(add_agent_btn)
        toolbar.addStretch()
        main_layout.addLayout(toolbar)

        # Grid for agents
        self.grid = QGridLayout()
        main_layout.addLayout(self.grid)

        self.agent_count = 0
        self.max_columns = 3

    def add_agent(self):
        row = self.agent_count // self.max_columns
        col = self.agent_count % self.max_columns

        agent = AgentWidget()
        self.grid.addWidget(agent, row, col)
        self.agent_count += 1


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()