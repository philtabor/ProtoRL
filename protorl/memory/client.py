import uuid


class MemoryClient:
    def __init__(self, request_queue, response_queue):
        self.request_queue = request_queue
        self.response_queue = response_queue

    def _send_request(self, request):
        request_id = str(uuid.uuid4())
        self.request_queue.put((request_id, request))
        while True:
            resp_id, response = self.response_queue.get()
            if resp_id == request_id:
                return response
            else:
                self.response_queue.put((resp_id, response))

    def add(self, experience, vals=None):
        return self._send_request(('add', experience, vals))

    def sample(self):
        return self._send_request(('sample', ))

    def update_priorities(self, indices, priorities):
        return self._send_request(('update', indices, priorities))

    def ready(self):
        return self._send_request(('ready',))

    def exit(self):
        return self._send_request(('exit',))
