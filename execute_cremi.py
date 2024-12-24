import subprocess
import sys

def main():
    valid_options = [0, 1, 2, 3]

    try:
        number = int(sys.argv[1])
        if number not in valid_options:
            print(f"Invalid choice. Please select one of {valid_options}.")
            sys.exit(1)
    except ValueError:
        print("Invalid input.")
        sys.exit(1)

    # Define the order of execution
    execution_order = {
        0: ['00', '01', '10', '11'],
        1: ['01', '10', '11', '00'],
        2: ['10', '11', '00', '01'],
        3: ['11', '00', '01', '10']
    }

    # Get the execution order based on the input number
    order = execution_order[number]

    # Execute commands for each number in the order
    for num in order:
        commands = [
            f"python3 main.py configs/conf_{num}0.yaml",
            f"python3 main.py configs/conf_{num}1.yaml"
        ]

        for command in commands:
            print(f"Executing: {command}")
            process = subprocess.run(command, shell=True)
            if process.returncode != 0:
                print(f"Command failed: {command}")
                sys.exit(1)

    print("All commands executed successfully.")

if __name__ == "__main__":
    main()