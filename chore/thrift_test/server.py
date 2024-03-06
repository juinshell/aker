from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
import sys
sys.path.append('gen-py')

from expr import AargsPingService
from expr.ttypes import Request, Response

class Handler:
    def ping(self, request):
        print("Received request:")
        print("Request ID:", request.request_id)
        print("Workflow Name:", request.workflow_name)
        if request.input_datas:
            for k, v in request.input_datas.input_data.items():
                print("Input Data:", k, v)

        # 构造响应
        response = Response(request_id=request.request_id, status="OK")
        return response

handler = Handler()
processor = AargsPingService.Processor(handler)

transport = TSocket.TServerSocket(port=9090)
tfactory = TTransport.TBufferedTransportFactory()
pfactory = TBinaryProtocol.TBinaryProtocolFactory()

server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
print("Starting the server...")
server.serve()
