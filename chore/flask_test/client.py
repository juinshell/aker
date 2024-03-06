from gevent import monkey

monkey.patch_all()
import time
import gevent

from typing import Dict


# this information is different per request
class RequestInfo:
    def __init__(self, workflow_name, ips: set, templates_infos: Dict[str, dict]):
        # This request will cover how many nodes, it is useful for delete already finished RequestInfo in each node
        # after last function finished.
        self.workflow_name = workflow_name
        self.ips = ips
        self.templates_infos = templates_infos  # including the pre-allocate IP address of each function
        self.templates_infos['VIRTUAL'] = {'ip': '127.0.0.1'}


import requests

base_url = 'http://127.0.0.1:{}/{}'

times = []
def post_request(request_id):
    request_info = {'request_id': request_id,
                    'workflow_name': 'file_processing',
                    'input_datas': {'$USER.read_file_address': {'datatype': 'entity', 'val': '/text/sample.md'},
                                    '$USER.upload_address': {'datatype': 'entity', 'val': '127.0.0.1'}}}
    st = time.perf_counter()
    r = requests.post(base_url.format(7999, 'run'), json=request_info)
    ed = time.perf_counter()
    print(ed - st, r.json())
    times.append(ed - st)

qps_lists = [1, 10, 100, 1000]
events = []
for qps in qps_lists:
    sleep_time = 1 / qps
    for i in range(100):
        events.append(gevent.spawn(post_request, 'request_' + str(i).rjust(3, '0')))
        gevent.sleep(sleep_time)
    for e in events:
        e.join()
    times = times[10:]
    print('QPS:', qps, 'Average time:', sum(times) * 1000 / len(times), "ms", file=open('client.log', 'a'))