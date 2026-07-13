import unittest

from ai.publishers.mqtt_identity import build_mqtt_client_id


class MqttClientIdentityTest(unittest.TestCase):
    def test_status_and_inference_for_same_camera_are_distinct(self):
        status = build_mqtt_client_id("strange-ai", "cam_02", "status", pid=101)
        inference = build_mqtt_client_id("strange-ai", "cam_02", "inference", pid=101)
        self.assertNotEqual(status, inference)

    def test_camera_role_and_pid_are_part_of_client_id(self):
        cam_02 = build_mqtt_client_id("strange-ai", "cam_02", "inference", pid=101)
        cam_03 = build_mqtt_client_id("strange-ai", "cam_03", "inference", pid=101)
        self.assertNotEqual(cam_02, cam_03)
        self.assertEqual(cam_02, "strange-ai-cam_02-inference-101")
