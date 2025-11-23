import json
import os


def convert_tsplib_jsons():
    input_folder = "datasets/tsplib/tests"
    output_folder = "resources/tsplib/input"

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if not file.endswith(".json"):
            continue
        
        input_path = os.path.join(input_folder, file)

        # Load original JSON file
        with open(input_path, "r") as f:
            data = json.load(f)

        # Extract needed fields
        n = data.get("n")
        k = data.get("k")
        optimal_solution = data.get("distance")
        x_coords = data.get("x")
        y_coords = data.get("y")

        y = file.split(".")
        name = y[0]

        # Build new format
        new_json = {
            "format": 2,
            "name": name,
            "n": n,
            "k": k,
            "optimal_solution": optimal_solution,
            "x_values": x_coords,
            "y_values": y_coords
        }

        # Save to output folder with same filename
        output_path = os.path.join(output_folder, file)
        with open(output_path, "w") as f:
            json.dump(new_json, f, indent=4)

        print(f"Converted {file} → {output_path}")


# Run the conversion
if __name__ == "__main__":
    convert_tsplib_jsons()
