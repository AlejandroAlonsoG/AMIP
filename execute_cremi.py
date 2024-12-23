import subprocess
import sys

def main():
    valid_options = ['00', '01', '10', '11']

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
        '00': ['00', '01', '10', '11'],
        '01': ['01', '10', '11', '00'],
        '10': ['10', '11', '00', '01'],
        '11': ['11', '00', '01', '10']
    }

    # Get the execution order based on the input number
    order = execution_order[number]

    # Execute commands for each number in the order
    for num in order:
        commands = [
            f"python3 main.py --cfg configs/conf_{num}0.yaml",
            f"python3 main.py --cfg configs/conf_{num}1.yaml"
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