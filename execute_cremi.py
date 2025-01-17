import subprocess
import sys

def main():
    valid_options = [0, 1, 2, 3, 4, 5, 6, 7]

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
        0: ['000', '001'],
        1: ['001', '010'],
        2: ['010', '011'],
        3: ['011', '100'],
        4: ['100', '101'],
        5: ['101', '110'],
        6: ['110', '111'],
        7: ['111', '000']
    }

    # Get the execution order based on the input number
    order = execution_order[number]

    # Execute commands for each number in the order
    for num in order:
        commands = [
            f"python3 main.py configs/conf_{num}.yaml",
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