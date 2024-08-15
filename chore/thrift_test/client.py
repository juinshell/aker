import time
from gevent import monkey

monkey.patch_all()
import gevent
from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
import sys
sys.path.append('gen-py')
from expr import AargsPingService
from expr.ttypes import Request, Response, InputDatas

# 创建 Thrift 传输和协议对象
transport = TSocket.TSocket('localhost', 9090)
transport = TTransport.TBufferedTransport(transport)
protocol = TBinaryProtocol.TBinaryProtocol(transport)

# 创建 Thrift 客户端
client = AargsPingService.Client(protocol)

# 打开传输
transport.open()
times = []
def post_request(request_id):
    # 构造请求
    request = Request(request_id=int(request_id), workflow_name="example", input_datas=InputDatas({'$USER.read_file_address': {'datatype': 'entity', 'val': '/text/sample.md'},
                                    '$USER.upload_address': {'datatype': 'entity', 'val': '127.0.0.1'}}))
    st = time.perf_counter()
    response = client.ping(request)
    ed = time.perf_counter()
    print(ed - st, response.status)
    times.append(ed - st)
# 调用服务端的 ping 方法

qps_lists = [1, 10, 100, 1000]
events = []
for qps in qps_lists:
    sleep_time = 1 / qps
    for i in range(100):
        events.append(gevent.spawn(post_request, 1))
        gevent.sleep(sleep_time)
    for e in events:
        e.join()
    times = times[10:]
    print('QPS:', qps, 'Average time:', sum(times) * 1000 / len(times), "ms", file=open('client.log', 'a'))

# 关闭传输
transport.close()
