#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from http.server import HTTPServer
from http.server import BaseHTTPRequestHandler
from http import HTTPStatus
import platform
import random

class MyHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-type', 'text/json')
        self.end_headers()
        hostname = platform.node()
        self.wfile.write('{{"name":"{}"}}'.format(hostname).encode('utf-8'))
        return


def run(server_class=HTTPServer, handler_class=MyHandler):
    server_address = ('localhost', 4040)
    httpd = server_class(server_address, handler_class)
    try:
        print("listening...")
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.socket.close()


if __name__ == '__main__':
    run()