# Local MQTT Broker

This EMQX broker is for local AI-to-backend event validation before AWS deployment.

## Start

```bash
docker compose -f infra/local-mqtt/docker-compose.yml up -d
```

Fallback:

```bash
docker-compose -f infra/local-mqtt/docker-compose.yml up -d
```

MQTT listens on `localhost:1883`.
The EMQX Dashboard listens on `http://localhost:18083`.

## Stop

```bash
docker compose -f infra/local-mqtt/docker-compose.yml down
```

## Local Topic

Use `safety/events` for local backend subscription tests.

AWS deployment should inject the broker endpoint through environment variables instead of changing code.

## Environment

```text
MQTT_HOST=localhost
MQTT_PORT=1883
MQTT_TOPIC=safety/events
MQTT_CLIENT_ID=strange-ai-local
MQTT_USERNAME=
MQTT_PASSWORD=
```

## Publish Test Event

```bash
python scripts/publish_test_mqtt_event.py
```

## Subscribe

```bash
mosquitto_sub -h localhost -p 1883 -t safety/events
```

Or:

```bash
python scripts/subscribe_mqtt_events.py
```

## Spring Boot Local Settings

```yaml
mqtt:
  host: localhost
  port: 1883
  topic: safety/events
```
