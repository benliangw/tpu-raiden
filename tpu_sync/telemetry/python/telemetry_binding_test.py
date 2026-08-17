# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for TPU Raiden Python telemetry bindings."""

from absl.testing import absltest
from tpu_sync.telemetry.python import _telemetry_binding_test_ext as telemetry_ext


class TelemetryBindingTest(absltest.TestCase):

  def tearDown(self):
    super().tearDown()
    telemetry_ext.configure_telemetry([])

  def test_configure_telemetry_enable_prometheus(self):
    telemetry_ext.configure_telemetry(["prometheus"])
    snapshot = telemetry_ext.get_raiden_metrics_prometheus_text()
    self.assertIn(
        "# HELP tpu_raiden_sent_bytes_total Total count of bytes sent over TPU"
        " Raiden interfaces.",
        snapshot,
    )
    self.assertIn("# TYPE tpu_raiden_sent_bytes_total counter", snapshot)
    self.assertIn(
        "# HELP tpu_raiden_received_bytes_total Total count of bytes received"
        " over TPU Raiden interfaces.",
        snapshot,
    )
    self.assertIn("# TYPE tpu_raiden_received_bytes_total counter", snapshot)
    self.assertIn(
        "# HELP tpu_raiden_transfer_failures_total Cumulative total count of"
        " transfer failures across all interfaces.",
        snapshot,
    )
    self.assertIn("# TYPE tpu_raiden_transfer_failures_total counter", snapshot)

  def test_configure_telemetry_case_insensitive(self):
    telemetry_ext.configure_telemetry(["Prometheus"])
    snapshot = telemetry_ext.get_raiden_metrics_prometheus_text()
    self.assertIn(
        "# HELP tpu_raiden_sent_bytes_total Total count of bytes sent over TPU"
        " Raiden interfaces.",
        snapshot,
    )
    self.assertIn("# TYPE tpu_raiden_sent_bytes_total counter", snapshot)

  def test_configure_telemetry_duplicate_backends_deduplicated(self):
    telemetry_ext.configure_telemetry(["prometheus", "prometheus"])
    snapshot = telemetry_ext.get_raiden_metrics_prometheus_text()
    self.assertEqual(
        snapshot.count(
            "# HELP tpu_raiden_sent_bytes_total Total count of bytes sent over"
            " TPU Raiden interfaces."
        ),
        1,
    )
    self.assertEqual(
        snapshot.count("# TYPE tpu_raiden_sent_bytes_total counter"), 1
    )

  def test_configure_telemetry_empty_backends_clears_backends(self):
    telemetry_ext.configure_telemetry(["prometheus"])
    telemetry_ext.configure_telemetry([])
    snapshot = telemetry_ext.get_raiden_metrics_prometheus_text()
    self.assertEqual(snapshot, "")

  def test_configure_telemetry_unknown_backend_raises_value_error(self):
    with self.assertRaisesRegex(
        ValueError, "Unknown telemetry backend: unknown_backend"
    ):
      telemetry_ext.configure_telemetry(["unknown_backend"])

  def test_configure_telemetry_tuple_sequence_supported(self):
    telemetry_ext.configure_telemetry(("prometheus",))
    snapshot = telemetry_ext.get_raiden_metrics_prometheus_text()
    self.assertIn("# TYPE tpu_raiden_sent_bytes_total counter", snapshot)

  def test_configure_telemetry_set_raises_type_error(self):
    with self.assertRaises(TypeError):
      telemetry_ext.configure_telemetry({"prometheus"})

  def test_configure_telemetry_invalid_argument_type_raises_type_error(self):
    with self.assertRaises(TypeError):
      telemetry_ext.configure_telemetry(123)
    with self.assertRaises(TypeError):
      telemetry_ext.configure_telemetry([123])
    with self.assertRaises(TypeError):
      telemetry_ext.configure_telemetry(None)

  def test_metric_type_enum(self):
    self.assertTrue(hasattr(telemetry_ext.MetricType, "COUNTER"))
    self.assertTrue(hasattr(telemetry_ext.MetricType, "GAUGE"))
    self.assertTrue(hasattr(telemetry_ext.MetricType, "HISTOGRAM"))
    self.assertNotEqual(
        telemetry_ext.MetricType.COUNTER, telemetry_ext.MetricType.GAUGE
    )
    self.assertNotEqual(
        telemetry_ext.MetricType.COUNTER, telemetry_ext.MetricType.HISTOGRAM
    )
    self.assertNotEqual(
        telemetry_ext.MetricType.GAUGE, telemetry_ext.MetricType.HISTOGRAM
    )

  def test_metric_metadata_properties_repr_equality(self):
    self.assertTrue(hasattr(telemetry_ext, "ALL_METRICS"))
    metrics = telemetry_ext.ALL_METRICS
    self.assertGreaterEqual(len(metrics), 3)

    sent_bytes_meta = metrics[0]
    self.assertEqual(sent_bytes_meta.name, "sent_bytes_total")
    self.assertIn("sent", sent_bytes_meta.description.lower())
    self.assertEqual(
        sent_bytes_meta.buckets,
        [
            0.1,
            0.25,
            0.5,
            1.0,
            2.5,
            5.0,
            10.0,
            25.0,
            50.0,
            100.0,
            250.0,
            500.0,
            750.0,
            1000.0,
            2500.0,
            5000.0,
            7500.0,
            10000.0,
            25000.0,
            50000.0,
        ],
    )

    # Test __repr__
    repr_str = repr(sent_bytes_meta)
    self.assertIn("sent_bytes_total", repr_str)
    self.assertIn("MetricType.COUNTER", repr_str)

    # Test equality with same object / identical values
    self.assertEqual(sent_bytes_meta, metrics[0])
    self.assertNotEqual(sent_bytes_meta, metrics[1])

    # Test heterogeneous equality comparisons (must not raise TypeError)
    self.assertIsNotNone(sent_bytes_meta)
    self.assertNotEqual(sent_bytes_meta, "sent_bytes_total")
    self.assertNotEqual(sent_bytes_meta, 42)

  def test_get_metric_metadata_empty_when_no_backends(self):
    telemetry_ext.configure_telemetry([])
    metadata = telemetry_ext.get_metric_metadata()
    self.assertEqual(metadata, [])

  def test_get_metric_metadata_with_prometheus(self):
    telemetry_ext.configure_telemetry(["prometheus"])
    metadata = telemetry_ext.get_metric_metadata()
    self.assertIsInstance(metadata, list)
    self.assertEqual(metadata, telemetry_ext.ALL_METRICS)

  def test_get_and_reset_metric_samples_empty_when_no_backends(self):
    telemetry_ext.configure_telemetry([])
    samples = telemetry_ext.get_and_reset_metric_samples()
    self.assertEqual(samples, {})

  def test_get_and_reset_metric_samples_with_prometheus(self):
    telemetry_ext.configure_telemetry(["prometheus"])
    samples = telemetry_ext.get_and_reset_metric_samples()
    self.assertEqual(samples, {})

  def test_configure_telemetry_buffered(self):
    telemetry_ext.configure_telemetry(["buffered"])
    metadata = telemetry_ext.get_metric_metadata()
    self.assertEqual(metadata, telemetry_ext.ALL_METRICS)
    samples = telemetry_ext.get_and_reset_metric_samples()
    self.assertEqual(samples, {})


if __name__ == "__main__":
  absltest.main()
