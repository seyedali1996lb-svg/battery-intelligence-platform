"""
Synthetic fault-injection harness for the MQTT ingestion pipeline
(src/mqtt_stream.py).

Everything in this package is SYNTHETIC or REPLAYED-PUBLIC-DATASET
traffic: it takes already-cached, real NASA PCoE cycling data (see
batlab/datasets/nasa.py -- reads pre-committed CSVs under data/raw/, no
network call) and deliberately corrupts it (timestamps, units, dropped
packets) before replaying it through the real ingestion function
(mqtt_stream._on_message()). No live BMS hardware or partner account is
involved anywhere here, and this harness must never be described as
validating anything against real industrial data -- it proves the
ingestion layer's fault-detection logic (src/mqtt_stream.py's
validate_telemetry()) catches corrupted data before any real partner
integration exists to catch it for real.
"""
