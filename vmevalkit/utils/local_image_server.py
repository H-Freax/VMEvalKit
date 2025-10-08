"""
Simple local HTTP server to serve images for Luma API.

This is a temporary solution to provide public URLs for local images
without using S3 or external services.
"""

import http.server
import socketserver
import threading
import os
import time
from pathlib import Path
from typing import Optional, Dict
import socket


class ImageServer:
    """
    Simple HTTP server to serve local images temporarily.
    
    This allows Luma to access our local images via HTTP URLs.
    """
    
    def __init__(self, port: int = 0, directory: str = "."):
        """
        Initialize the image server.
        
        Args:
            port: Port to serve on (0 = auto-select available port)
            directory: Directory to serve files from
        """
        self.directory = Path(directory).resolve()
        self.port = port
        self.server: Optional[socketserver.TCPServer] = None
        self.thread: Optional[threading.Thread] = None
        self.hostname = self._get_public_ip()
        
    def _get_public_ip(self) -> str:
        """Get the public IP address of this machine."""
        try:
            # Try to get external IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            # Fallback to localhost
            return "localhost"
    
    def start(self) -> int:
        """
        Start the HTTP server in a background thread.
        
        Returns:
            The port number the server is running on
        """
        # Create a simple HTTP request handler
        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(self.server.directory), **kwargs)
            
            def log_message(self, format, *args):
                # Suppress logging
                pass
        
        # Bind the handler to our directory
        Handler.directory = self.directory
        
        # Create server with auto-selected port if needed
        self.server = socketserver.TCPServer(("", self.port), Handler)
        if self.port == 0:
            self.port = self.server.server_address[1]
        
        # Start server in background thread
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()
        
        # Give it a moment to start
        time.sleep(0.1)
        
        print(f"[ImageServer] Started on http://{self.hostname}:{self.port}")
        return self.port
    
    def stop(self):
        """Stop the HTTP server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            if self.thread:
                self.thread.join(timeout=1)
            print("[ImageServer] Stopped")
    
    def get_url(self, file_path: str) -> str:
        """
        Get the HTTP URL for a local file.
        
        Args:
            file_path: Path to the file (relative to server directory)
            
        Returns:
            HTTP URL for the file
        """
        # Make path relative to server directory
        try:
            rel_path = Path(file_path).relative_to(self.directory)
        except ValueError:
            # If not relative, just use the filename
            rel_path = Path(file_path).name
        
        return f"http://{self.hostname}:{self.port}/{rel_path}"


# Singleton instance
_server_instance: Optional[ImageServer] = None


def get_image_server(directory: str = ".") -> ImageServer:
    """Get or create the singleton image server."""
    global _server_instance
    if _server_instance is None:
        _server_instance = ImageServer(directory=directory)
        _server_instance.start()
    return _server_instance


def stop_image_server():
    """Stop the singleton image server."""
    global _server_instance
    if _server_instance:
        _server_instance.stop()
        _server_instance = None
