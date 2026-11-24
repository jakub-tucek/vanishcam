import http.client
import threading
import time
import unittest

import cv2
import numpy as np

from browser_output import BrowserOutput

MEET = "https://meet.google.com"


class BrowserOutputTests(unittest.TestCase):
    def setUp(self):
        self.output = BrowserOutput(port=0)
        self.addCleanup(self.output.close)

    def request(self, method="GET", path="/frame?after=0", origin=MEET, client=True, host=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.output.port, timeout=5)
        headers = {"Host": host or f"127.0.0.1:{self.output.port}"}
        if origin:
            headers["Origin"] = origin
        if client:
            headers["X-VanishCam"] = "1"
        connection.request(method, path, headers=headers)
        response = connection.getresponse()
        body = response.read()
        connection.close()
        return response, body

    def publish_while_polling(self, frame):
        stop = threading.Event()

        def publisher():
            while not stop.is_set():
                self.output.publish(frame)
                time.sleep(0.02)

        thread = threading.Thread(target=publisher)
        thread.start()
        self.addCleanup(lambda: (stop.set(), thread.join()))

    def test_meet_receives_jpeg_frames(self):
        frame = np.full((48, 64, 3), (10, 120, 220), np.uint8)
        self.publish_while_polling(frame)
        response, body = self.request()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Access-Control-Allow-Origin"), MEET)
        decoded = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, frame.shape)
        self.assertLess(np.abs(decoded.astype(int) - frame).max(), 8)
        self.assertTrue(self.output.connected)

    def test_other_websites_and_rebinding_are_refused(self):
        self.publish_while_polling(np.zeros((8, 8, 3), np.uint8))
        for options in (dict(origin="https://evil.example"), dict(origin=None),
                        dict(client=False), dict(host="attacker.example:47809"),
                        dict(path="/other")):
            with self.subTest(**options):
                response, _ = self.request(**options)
                self.assertEqual(response.status, 403)

    def test_preflight_allows_only_meet(self):
        response, _ = self.request("OPTIONS", client=False)
        self.assertEqual(response.status, 204)
        self.assertEqual(response.getheader("Access-Control-Allow-Headers"), "X-VanishCam")
        self.assertEqual(response.getheader("Access-Control-Allow-Private-Network"), "true")
        response, _ = self.request("OPTIONS", origin="https://evil.example", client=False)
        self.assertEqual(response.status, 403)

    def test_stale_frame_id_after_app_restart_still_gets_frames(self):
        self.publish_while_polling(np.zeros((8, 8, 3), np.uint8))
        response, _ = self.request(path="/frame?after=999999")
        self.assertEqual(response.status, 200)

    def test_idle_until_a_client_polls(self):
        self.output.publish(np.zeros((8, 8, 3), np.uint8))
        self.assertFalse(self.output.connected)
        response, _ = self.request()
        self.assertEqual(response.status, 204)  # Nothing new within the wait window.


if __name__ == "__main__":
    unittest.main()
