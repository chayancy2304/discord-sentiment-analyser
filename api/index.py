from http.server import BaseHTTPRequestHandler
import json


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
            self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                            self.end_headers()
                                    response = json.dumps({"status": "ok", "message": "Discord Sentiment Analyser bot is running."})
                                            self.wfile.write(response.encode())

                                                def do_POST(self):
                                                        self.send_response(200)
                                                                self.send_header('Content-type', 'application/json')
                                                                        self.end_headers()
                                                                                response = json.dumps({"status": "ok"})
                                                                                        self.wfile.write(response.encode())
