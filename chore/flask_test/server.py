import os
import subprocess
import sys
import time
import uuid
import yaml
from typing import List

sys.path.append('../../')
from flask import Flask, request

proxy = Flask(__name__)

@proxy.route('/run', methods=['post'])
def start():
    data = request.get_json(force=True, silent=True)
    print(data)
    ret = {
        "request_id": data['request_id'],
        "status": "OK",
    }
    return data, 200

if __name__ == '__main__':
    proxy.run('0.0.0.0', 7999, threaded=True)