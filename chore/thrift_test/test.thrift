namespace py expr
service PingService {
    string ping(),
}

struct Request {
    1: required i32 request_id,
    2: required string workflow_name,
    3: optional InputDatas input_datas
}

struct InputDatas {
    3: optional map<string, map<string, string>> input_data
}

struct Response {
    1: required i32 request_id,
    2: required string status
}

service AargsPingService {
    Response ping(1: Request request),
}