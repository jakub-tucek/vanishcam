import threading
import time
import unittest

from gesture_worker import GestureJob, GestureWorker, Observation


def job(now=0.0, modes=("snap",)):
    return GestureJob(image=None, width=1, height=1, now=now, phase="ready", modes=modes,
                      recognizers={})


def wait_for(worker, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = worker.poll()
        if result is not None:
            return result
        time.sleep(0.005)
    raise AssertionError("worker produced no result")


class GestureWorkerTests(unittest.TestCase):
    def test_threaded_results_come_back_once(self):
        worker = GestureWorker(lambda j: [Observation(m, True, "") for m in j.modes])
        self.addCleanup(worker.close)
        self.assertTrue(worker.submit(job(1.0)))
        finished, observations = wait_for(worker)
        self.assertEqual(finished.now, 1.0)
        self.assertEqual([o.mode for o in observations], ["snap"])
        self.assertIsNone(worker.poll())

    def test_frames_are_skipped_while_busy(self):
        release = threading.Event()
        calls = []

        def slow(j):
            calls.append(j.now)
            release.wait(5)
            return []

        worker = GestureWorker(slow)
        self.addCleanup(worker.close)
        self.assertTrue(worker.submit(job(1.0)))
        self.assertFalse(worker.submit(job(2.0)))
        self.assertTrue(worker.busy)
        release.set()
        wait_for(worker)
        self.assertEqual(calls, [1.0])
        self.assertTrue(worker.submit(job(3.0)))
        wait_for(worker)
        self.assertEqual(calls, [1.0, 3.0])

    def test_a_failure_reaches_the_main_loop(self):
        def broken(j):
            raise RuntimeError("model crashed")

        worker = GestureWorker(broken)
        self.addCleanup(worker.close)
        worker.submit(job())
        with self.assertRaisesRegex(RuntimeError, "model crashed"):
            wait_for(worker)

    def test_close_stops_the_thread_and_refuses_new_jobs(self):
        worker = GestureWorker(lambda j: [])
        worker.close()
        self.assertFalse(worker._thread.is_alive())
        self.assertFalse(worker.submit(job()))

    def test_synchronous_mode_answers_immediately(self):
        worker = GestureWorker(lambda j: [Observation("clones", False, "")], threaded=False)
        worker.submit(job())
        self.assertEqual(worker.poll()[1][0].mode, "clones")


if __name__ == "__main__":
    unittest.main()
