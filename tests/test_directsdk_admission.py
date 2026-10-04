"""One upstream admission and first-response authority over native recovery."""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import directsdk

NATIVE = r'''
import json, os, sys, urllib.request, urllib.error
for line in sys.stdin:
    frame=json.loads(line)
    if frame.get('shouldQuery') is False:
        print(json.dumps({'type':'result','num_turns':0}),flush=True)
        continue
    break
url=os.environ['ANTHROPIC_BASE_URL']+'/v1/messages'
for _ in range(2):
    try:
        urllib.request.urlopen(urllib.request.Request(url,data=b'{}',headers={'Content-Type':'application/json'}),timeout=5).read()
    except urllib.error.HTTPError:
        break
print(json.dumps({'type':'assistant','message':{'id':'first','role':'assistant','content':[{'type':'text','text':'FIRST'}]}}))
print(json.dumps({'type':'stream_event','event':{'type':'message_stop'}}))
print(json.dumps({'type':'result','subtype':'success','usage':{'input_tokens':0,'output_tokens':0}}))
'''


@pytest.mark.parametrize('stop', ['end_turn', 'max_tokens', 'model_context_window_exceeded'])
def test_first_response_owns_usage_and_stops_recovery(tmp_path, stop):
    calls = []
    usage = {'input_tokens':0, 'output_tokens':0, 'cache_read_input_tokens':0, 'cache_creation_input_tokens':0}
    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            calls.append(self.path)
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.end_headers()
            events = [
                {'type':'message_start','message':{'id':'first','role':'assistant','model':'sonnet','content':[], 'usage':usage}},
                {'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}},
                {'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':'FIRST'}},
                {'type':'content_block_stop','index':0},
                {'type':'message_delta','delta':{'stop_reason':stop},'usage':usage},
                {'type':'message_stop'},
            ]
            self.wfile.write(''.join('data: '+json.dumps(e)+'\n\n' for e in events).encode())
    peer=ThreadingHTTPServer(('127.0.0.1',0),Peer)
    thread=threading.Thread(target=peer.serve_forever,daemon=True); thread.start()
    native=tmp_path/'native.py'; native.write_text(NATIVE)
    client=directsdk.Client(command=[sys.executable,str(native)],env={'PATH':os.defpath,'HOME':str(tmp_path),'ANTHROPIC_BASE_URL':f'http://127.0.0.1:{peer.server_port}'})
    try:
        result=client.create(model='sonnet',messages=[{'role':'user','content':'fixture'}])
        assert len(calls)==1
        assert result.choices[0].message.content=='FIRST'
        assert result.choices[0].finish_reason==('stop' if stop=='end_turn' else 'length')
        assert result.usage.prompt_tokens==0
        assert result.choices[0].message.reasoning_details[0]['messages'][0]['stop_reason']==stop
    finally:
        client.close(); peer.shutdown(); thread.join(); peer.server_close()


def test_empty_tool_input_completes_the_capture(tmp_path):
    """A no-argument tool call streams an empty input_json_delta; the capture must still complete."""
    calls = []
    usage = {'input_tokens':0, 'output_tokens':0, 'cache_read_input_tokens':0, 'cache_creation_input_tokens':0}
    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            calls.append(self.path)
            self.rfile.read(int(self.headers['Content-Length']))
            self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.end_headers()
            events = [
                {'type':'message_start','message':{'id':'first','role':'assistant','model':'sonnet','content':[], 'usage':usage}},
                {'type':'content_block_start','index':0,'content_block':{'type':'tool_use','id':'toolu_1','name':'mcp__hermes__list_things','input':{}}},
                {'type':'content_block_delta','index':0,'delta':{'type':'input_json_delta','partial_json':''}},
                {'type':'content_block_stop','index':0},
                {'type':'message_delta','delta':{'stop_reason':'tool_use'},'usage':usage},
                {'type':'message_stop'},
            ]
            self.wfile.write(''.join('data: '+json.dumps(e)+'\n\n' for e in events).encode())
    peer=ThreadingHTTPServer(('127.0.0.1',0),Peer)
    thread=threading.Thread(target=peer.serve_forever,daemon=True); thread.start()
    native=tmp_path/'native.py'; native.write_text(NATIVE)
    tools=[{'type':'function','function':{'name':'list_things','description':'list','parameters':{'type':'object','properties':{}}}}]
    client=directsdk.Client(command=[sys.executable,str(native)],env={'PATH':os.defpath,'HOME':str(tmp_path),'ANTHROPIC_BASE_URL':f'http://127.0.0.1:{peer.server_port}'})
    try:
        result=client.create(model='sonnet',messages=[{'role':'user','content':'fixture'}],tools=tools)
        assert len(calls)==1
        call=result.choices[0].message.tool_calls[0]
        assert call.function.name=='list_things'
        assert json.loads(call.function.arguments)=={}
        assert result.choices[0].finish_reason=='tool_calls'
    finally:
        client.close(); peer.shutdown(); thread.join(); peer.server_close()


def test_cancel_closes_the_active_upstream_socket(tmp_path):
    entered, disconnected = threading.Event(), threading.Event()
    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            entered.set()
            self.rfile.read(1)
            disconnected.set()
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Peer)
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    native = tmp_path / 'native.py'
    native.write_text(NATIVE)
    client = directsdk.Client(command=[sys.executable, str(native)], env={'PATH':os.defpath, 'HOME':str(tmp_path), 'ANTHROPIC_BASE_URL':f'http://127.0.0.1:{peer.server_port}'})
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(client.create, model='sonnet', messages=[{'role':'user', 'content':'fixture'}])
            try:
                assert entered.wait(5)
            finally:
                client.cancel()
            with pytest.raises(RuntimeError, match='cancelled'):
                result.result(timeout=3)
            assert disconnected.wait(2)
    finally:
        client.close(); peer.shutdown(); thread.join(); peer.server_close()


def test_incomplete_upstream_error_names_the_first_attempt(tmp_path):
    """Native's retries are denied with ADMISSION_CONSUMED; the raised error must carry the first attempt's status."""
    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            body=b'{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 213000 tokens > 200000 maximum"}}'
            self.send_response(529); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
    peer=ThreadingHTTPServer(('127.0.0.1',0),Peer)
    thread=threading.Thread(target=peer.serve_forever,daemon=True); thread.start()
    native=tmp_path/'native.py'; native.write_text(NATIVE)
    client=directsdk.Client(command=[sys.executable,str(native)],env={'PATH':os.defpath,'HOME':str(tmp_path),'ANTHROPIC_BASE_URL':f'http://127.0.0.1:{peer.server_port}'})
    try:
        with pytest.raises(RuntimeError, match=r'status 529, capture incomplete.*upstream said: prompt is too long: 213000 tokens'):
            client.create(model='sonnet',messages=[{'role':'user','content':'fixture'}])
    finally:
        client.close(); peer.shutdown(); thread.join(); peer.server_close()


def test_invalid_stream_json_error_names_the_offending_line(tmp_path):
    """A native that prints a non-JSON stdout line (a shim banner) fails with that line in the error, not a bare label."""
    native=tmp_path/'native.py'; native.write_text("import sys\nprint('mise WARN tool not activated')\nsys.exit(0)\n")
    client=directsdk.Client(command=[sys.executable,str(native)],env={'PATH':os.defpath,'HOME':str(tmp_path),'ANTHROPIC_BASE_URL':'http://127.0.0.1:9'})
    try:
        with pytest.raises(RuntimeError, match=r"Invalid native stream-json output: 'mise WARN tool not activated"):
            client.create(model='sonnet',messages=[{'role':'user','content':'fixture'}])
    finally:
        client.close()


def test_abort_closes_sockets_where_shutdown_cannot_wake_a_blocked_recv(monkeypatch):
    """Windows: shutdown() does not wake a recv blocked in another thread (the relay waiting on a hung
    upstream), so close() would wait for the upstream; closesocket() cancels it. POSIX keeps shutdown only."""
    import admission
    calls = []
    class Sock:
        def shutdown(self, how):
            calls.append('shutdown')
        def detach(self):
            calls.append('detach')
            return -1  # no real handle behind the fake
    for windows, expected in ((False, ['shutdown']), (True, ['shutdown', 'detach'])):
        monkeypatch.setattr(admission, '_CANCEL_BY_CLOSE', windows, raising=False)
        gate = admission.Admission('https://api.anthropic.com', 5)
        try:
            calls.clear()
            gate.sockets.add(Sock())
            gate.abort()
            assert calls == expected
        finally:
            gate.sockets.clear()
            gate.close()



@pytest.mark.parametrize('close_flag', [False, True])
def test_abort_wakes_a_real_getresponse_blocked_on_a_silent_upstream(monkeypatch, close_flag):
    """A real http.client read (which holds makefile() refs, so socket.close() alone never reaches the OS)
    blocked on an upstream that accepts and never answers must end promptly after abort()."""
    import admission, http.client, socket, time
    monkeypatch.setattr(admission, '_CANCEL_BY_CLOSE', close_flag, raising=False)
    entered, target = threading.Event(), []
    real_readinto = socket.SocketIO.readinto
    def readinto(self, buffer):
        if target and self._sock is target[0]:
            entered.set()  # the reader is about to block in recv on the upstream socket
        return real_readinto(self, buffer)
    monkeypatch.setattr(socket.SocketIO, 'readinto', readinto)
    listener = socket.create_server(('127.0.0.1', 0))
    listener.settimeout(5)
    gate = admission.Admission('https://api.anthropic.com', 30)
    conn = http.client.HTTPConnection('127.0.0.1', listener.getsockname()[1], timeout=30)
    peer = reader = None
    try:
        conn.request('POST', '/v1/messages', b'{}')
        peer, _ = listener.accept()  # held open and silent: the upstream never answers
        target.append(conn.sock)
        gate.sockets.add(conn.sock)
        outcome = []
        def read():
            try:
                outcome.append(conn.getresponse())
            except Exception as error:
                outcome.append(error)
        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        assert entered.wait(5), 'the reader never reached the blocking read'
        time.sleep(.2)  # let it pass from readinto() into the recv syscall
        assert reader.is_alive() and not outcome, 'the upstream never answers; the read must still be blocked'
        started = time.monotonic()
        gate.abort()
        reader.join(3)
        assert not reader.is_alive() and time.monotonic() - started < 3
        assert outcome and isinstance(outcome[0], Exception)
    finally:
        gate.sockets.clear()
        gate.close()
        if peer is not None:
            peer.close()  # EOF wakes a reader the abort failed to wake
        conn.close()
        listener.close()
        if reader is not None:
            reader.join(5)
