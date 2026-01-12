import sys
import os
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QHBoxLayout, QLabel
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QTextCursor
from openai import OpenAI


class ChatWorker(QThread):
    """Worker thread to handle OpenAI API calls with streaming."""
    chunk_received = Signal(str)
    stream_finished = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, client, messages):
        super().__init__()
        self.client = client
        self.messages = messages

    def run(self):
        try:
            stream = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.messages,
                stream=True
            )
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    self.chunk_received.emit(content)
            self.stream_finished.emit(full_response)
        except Exception as e:
            self.error_occurred.emit(str(e))


class ConsoleWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Console - OpenAI Chat")
        self.setMinimumSize(600, 400)

        # Initialize OpenAI client
        self.client = OpenAI()

        # Conversation history
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

        # Track if we're currently streaming
        self.is_streaming = False

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Status label
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        # Output area
        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setPlaceholderText("Chat will appear here...")
        layout.addWidget(self.output_area)

        # Input area
        input_layout = QHBoxLayout()

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Enter your message...")
        self.input_field.returnPressed.connect(self.process_input)
        input_layout.addWidget(self.input_field)

        self.submit_button = QPushButton("Send")
        self.submit_button.clicked.connect(self.process_input)
        input_layout.addWidget(self.submit_button)

        layout.addLayout(input_layout)

        # Button row
        button_layout = QHBoxLayout()

        self.clear_button = QPushButton("Clear Chat")
        self.clear_button.clicked.connect(self.clear_output)
        button_layout.addWidget(self.clear_button)

        self.reset_button = QPushButton("Reset Conversation")
        self.reset_button.clicked.connect(self.reset_conversation)
        button_layout.addWidget(self.reset_button)

        layout.addLayout(button_layout)

        self.worker = None

    def process_input(self):
        """Process the input and send to OpenAI."""
        text = self.input_field.text().strip()
        if not text:
            return

        # Display user message
        self.output_area.append(f"<b>You:</b> {text}")
        self.input_field.clear()

        # Add to conversation history
        self.messages.append({"role": "user", "content": text})

        # Disable input while processing
        self.set_input_enabled(False)
        self.status_label.setText("Thinking...")

        # Prepare for streaming response
        self.output_area.append("<b>Assistant:</b> ")
        self.is_streaming = True

        # Start worker thread
        self.worker = ChatWorker(self.client, self.messages.copy())
        self.worker.chunk_received.connect(self.handle_chunk)
        self.worker.stream_finished.connect(self.handle_stream_finished)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.start()

    def handle_chunk(self, chunk):
        """Handle a streaming chunk from OpenAI."""
        cursor = self.output_area.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(chunk)
        self.output_area.setTextCursor(cursor)
        self.output_area.ensureCursorVisible()
        self.status_label.setText("Streaming...")

    def handle_stream_finished(self, full_response):
        """Handle the completion of the stream."""
        self.is_streaming = False
        self.output_area.append("")  # Add spacing

        # Add to conversation history
        self.messages.append({"role": "assistant", "content": full_response})

        self.set_input_enabled(True)
        self.status_label.setText("Ready")

    def handle_error(self, error):
        """Handle errors from the API call."""
        self.is_streaming = False
        self.output_area.append(f"<b style='color: red;'>Error:</b> {error}")
        self.output_area.append("")

        # Remove the failed user message from history
        if self.messages and self.messages[-1]["role"] == "user":
            self.messages.pop()

        self.set_input_enabled(True)
        self.status_label.setText("Error - Ready to retry")

    def set_input_enabled(self, enabled):
        """Enable or disable input controls."""
        self.input_field.setEnabled(enabled)
        self.submit_button.setEnabled(enabled)

    def clear_output(self):
        """Clear the output area."""
        self.output_area.clear()

    def reset_conversation(self):
        """Reset the conversation history."""
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        self.output_area.clear()
        self.output_area.append("<i>Conversation reset.</i>")
        self.output_area.append("")
        self.status_label.setText("Ready")


def main():
    app = QApplication(sys.argv)
    window = ConsoleWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
