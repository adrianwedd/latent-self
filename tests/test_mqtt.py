import types, sys, pathlib
from types import SimpleNamespace
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

sys.modules.setdefault('cv2', types.ModuleType('cv2'))
sys.modules.setdefault('torch', types.ModuleType('torch'))
sys.modules.setdefault('mediapipe', types.ModuleType('mediapipe'))

import services

class DummyMQTT:
    def __init__(self, client_id=None):
        self.published = []
        self.tls_config = None
        self.auth = None

    def connect(self, host, port, keepalive):
        pass

    def loop_start(self):
        pass

    def publish(self, topic, payload):
        self.published.append((topic, payload))

    def disconnect(self):
        pass

    def loop_stop(self):
        pass

    def tls_set(self, ca_certs=None, certfile=None, keyfile=None):
        self.tls_config = (ca_certs, certfile, keyfile)

    def username_pw_set(self, username, password=None):
        self.auth = (username, password)


def test_heartbeat_interval_and_payload(monkeypatch):
    monkeypatch.setattr(services, 'MQTT_AVAILABLE', True)
    monkeypatch.setattr(services, 'mqtt', SimpleNamespace(Client=DummyMQTT))
    monkeypatch.setattr(services, 'platform', SimpleNamespace(node=lambda: 'testnode'))

    config = SimpleNamespace(data={'mqtt': {
        'enabled': True,
        'broker': 'localhost',
        'port': 1883,
        'topic_namespace': 'mirror',
        'device_id': 'dev1',
        'heartbeat_interval': 5,
    }})
    client = services.TelemetryClient(config)

    # first heartbeat
    monkeypatch.setattr(services, 'time', lambda: 10)
    client._last_ts = 0
    client.send_heartbeat()
    assert client.client.published
    topic, payload = client.client.published[-1]
    assert topic == 'mirror/dev1/heartbeat'
    assert 'alive' in payload

    # within interval -> no publish
    monkeypatch.setattr(services, 'time', lambda: 12)
    client.send_heartbeat()
    assert len(client.client.published) == 1

    # after interval -> publish again
    monkeypatch.setattr(services, 'time', lambda: 16)
    client.send_heartbeat()
    assert len(client.client.published) == 2


def test_tls_and_auth(monkeypatch):
    monkeypatch.setattr(services, 'MQTT_AVAILABLE', True)
    monkeypatch.setattr(services, 'mqtt', SimpleNamespace(Client=DummyMQTT))
    monkeypatch.setattr(services, 'platform', SimpleNamespace(node=lambda: 'n'))
    monkeypatch.setenv('PASS', 'secret')

    cfg = {
        'enabled': True,
        'broker': 'mqtts://broker.example.com:8883',
        'port': 8883,
        'topic_namespace': 'mirror',
        'device_id': 'dev1',
        'heartbeat_interval': 5,
        'username': 'user',
        'password': '$PASS',
        'ca_cert': '/ca.crt',
        'client_cert': '/client.crt',
        'client_key': '/client.key',
    }

    config = SimpleNamespace(data={'mqtt': cfg})
    client = services.TelemetryClient(config)

    assert client.client.auth == ('user', 'secret')
    assert client.client.tls_config == ('/ca.crt', '/client.crt', '/client.key')

