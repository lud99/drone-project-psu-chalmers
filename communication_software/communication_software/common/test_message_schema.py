import json_schemas

import json
import re


def read_json_with_comments(file_path):
    with open(file_path, "r") as f:
        content = f.read()

        # Remove single-line comments starting with //
        # This regex looks for // and grabs everything until the end of the line
        clean_content = re.sub(r"//.*", "", content)

        # Now it's safe to load
        return clean_content


try:
    data_string = read_json_with_comments("./test_app_backend_interface.jsonc")

    for example in json.loads(data_string):
        try:
            validated = json_schemas.parse_drone_message(json.dumps(example))
            print(f"Successfully parsed: {validated.msg_type}")

            # Example of handling different types
            if isinstance(validated, json_schemas.TelemetryMessage):
                print(f" -> Drone is at {validated.lat}, {validated.lon}")
            elif isinstance(validated, json_schemas.TaskMessage):
                print(f" -> New task: {validated.task.action}")

        except Exception as e:
            print(f"Validation failed for a message: {e}")

    data_string = read_json_with_comments("./test_telemetry_schema.jsonc")

    for example in json.loads(data_string):
        try:
            json_schemas.parse_telemetry(json.dumps(example))
            print("Successfully parsed telemetry")

        except Exception as e:
            print(f"Validation failed for telemetry: {e}")

    data_string = read_json_with_comments("./test_capabilities_schema.jsonc")

    for example in json.loads(data_string):
        try:
            json_schemas.parse_capabilities(json.dumps(example))
            print("Successfully parsed capability")

        except Exception as e:
            print(f"Validation failed for capability: {e}")

    data_string = read_json_with_comments("./test_detections_schema.jsonc")

    try:
        json_schemas.parse_detections(json.dumps(data_string))
        print("Successfully parsed decection")

    except Exception as e:
        print(f"Validation failed for decection: {e}")

except Exception as e:
    print(f"File loading error: {e}")
